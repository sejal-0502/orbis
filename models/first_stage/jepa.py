import math
import random
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from util import instantiate_from_config

class Jepa(pl.LightningModule):
    """
    Predictive SSL Model

    context_encoder : masked i/p img
    target encoder : fully visible i/p img
    predictor : i/p context encoder representation at mask positions, pos embeds for those mask positions
    loss : l1 loss (predicted_masks, traget tokens at mask indices)

    """
    
    def __init__(
        self,
        ema_momentum,
        warmup_steps,
        beta_1,
        beta_2,
        context_encoder_config,
        target_encoder_config,
        predictor_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__()

        ignore_keys = ignore_keys or []
        self.automatic_optimization = False
        self.grad_acc_steps = grad_acc_steps
        self.monitor = monitor
        # self.distill_model_type = distill_model_type
        self.cont_ratio_trainig = cont_ratio_trainig
        self.only_decoder=only_decoder
        self.min_lr_multiplier = min_lr_multiplier
        self.ema_momentum = ema_momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.warmup_steps = warmup_steps
        self.gan_loss = None
        
        assert (not scale_equivariance) or len(scale_equivariance) == 2, "if defined, scale_equivariance should be a list of two lists"
        self.scale_equivariance = scale_equivariance

        # Instantiate core components
        self.context_encoder = instantiate_from_config(context_encoder_config)
        self.target_encoder = instantiate_from_config(target_encoder_config)
        self.predictor_encoder = instantiate_from_config(predictor_config)

        # projection for feature comparison : [768 -> 384]
        self.pred_proj = nn.Linear(context_encoder_config.params["z_channels"], predictor_config.params["predictor_embed_dim"], 1)
        self.target_proj = nn.Linear(target_encoder_config.params["z_channels"], predictor_config.params["predictor_embed_dim"], 1)

        self.encoder_normalize_embedding = context_encoder_config.params.get("normalize_embedding", False)
        self.predictor_normalize_embedding = predictor_config.params.get("normalize_embedding", False)
        
        # Image and patch size
        self.image_size = context_encoder_config.params["resolution"]
        self.patch_size = context_encoder_config.params["patch_size"]

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        for t in self.target_encoder.parameters():
            t.requires_grad = False

    @staticmethod
    def _compute_scaled_size(image_size, patch_size):
        if isinstance(image_size, int):
            return [image_size * 14 // patch_size] * 2
        return [image_size[0] * 14 // patch_size, image_size[1] * 14 // patch_size]
    
    def get_input(self, batch):
        x = batch
        return x.float()

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def context_encode(self, x):
        h, mask_indices = self.context_encoder(x)

        B = h.shape[0]
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1)

        mask_tokens = h[batch_idx, mask_indices]
        pos = self.context_encoder.encoder.pos_embed # [N, D]
        pos = pos.squeeze(0)
        mask_pos_embed = pos[mask_indices]

        if self.encoder_normalize_embedding:
            mask_tokens = F.normalize(mask_tokens, p=2, dim=2) # [B, N, D]
        
        return mask_indices, mask_tokens, mask_pos_embed
    
    def target_encode(self, x, masked_indices):
        h, mask_indices = self.target_encoder(x)

        B = h.shape[0]
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1)
        
        mask_tokens = h[batch_idx, masked_indices]
        mask_tokens = self.target_proj(mask_tokens)

        if self.encoder_normalize_embedding:
            mask_tokens = F.normalize(mask_tokens, p=2, dim=2)
        
        return mask_tokens
    
    def predictor_encode(self, mask_tokens, mask_pos_embed):
        mask_tokens = self.pred_proj(mask_tokens)
        mask_pos_embed = self.pred_proj(mask_pos_embed)
        
        pred_tokens = self.predictor_encoder(mask_tokens, mask_pos_embed)
        
        if self.predictor_normalize_embedding:
            pred_tokens = F.normalize(pred_tokens, dim=2)

        return pred_tokens
        
    def forward(self, input):
        c_mask_indices, c_mask_tokens, c_mask_pos_embed = self.context_encode(input)

        with torch.no_grad():
            t_mask_tokens = self.target_encode(input, c_mask_indices)

        predicted_masks = self.predictor_encode(c_mask_tokens, c_mask_pos_embed)

        return t_mask_tokens, predicted_masks
    
    def l1_loss(self, pred_tokens, t_tokens):
        return F.l1_loss(pred_tokens, t_tokens)

    @torch.no_grad()
    def ema_update(self):
        context_params = dict(self.context_encoder.named_parameters())
        skipped_layers = 0
        updated_layers = 0
        prefix_to_strip = 'context_encoder.'

        for target_name, p_t in self.target_encoder.named_parameters():
            if target_name.startswith(prefix_to_strip):
                context_name = target_name[len(prefix_to_strip):]
            else:
                context_name = target_name

            if context_name in context_params:
                p_s = context_params[context_name]
                if p_s.shape == p_t.shape:
                    p_t.data = self.ema_momentum * p_t.data + (1 - self.ema_momentum) * p_s.data
                    updated_layers += 1
                else:
                    print(f"EMA Skipped: Shape mismatch for key '{context_name}' ({p_s.shape} vs {p_t.shape})")
                    skipped_layers += 1
            else:
                print(f"EMA Skipped: Parameter '{context_name}' not found in student encoder.")
                skipped_layers += 1

        # print(f">> Updated {updated_layers} parameter tensors.")
        # print(f"!! Skipped {skipped_layers} parameter tensors (due to missing key or shape mismatch).")


    def get_warmup_scheduler(self, optimizer, warmup_steps, min_lr_multiplier):
        min_lr = self.learning_rate * min_lr_multiplier
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step/warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = (1 - min_lr) * cosine_decay + min_lr
                return decayed
        
        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(list(self.context_encoder.parameters())+
                                  list(self.predictor_encoder.parameters()),
                                  lr=lr, betas=(self.beta_1, self.beta_2))
        
        scheduler_optim_warmup = self.get_warmup_scheduler(optimizer, self.warmup_steps, self.min_lr_multiplier)
        
        return [optimizer], [scheduler_optim_warmup]
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        target_masks, pred_masks = self(x)

        l1_loss = self.l1_loss(pred_masks, target_masks)

        self.log("val/l1_loss", l1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)


    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        schedule_optim_warmup = self.lr_schedulers()
        x = self.get_input(batch)
        
        target_masks, pred_masks = self(x)

        l1_loss = self.l1_loss(pred_masks, target_masks)

        self.log("train/l1_loss", l1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        self.manual_backward(l1_loss)
        if (batch_idx+1) % self.grad_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            schedule_optim_warmup.step()
            self.ema_update()