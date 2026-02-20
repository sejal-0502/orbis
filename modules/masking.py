import torch
import torch.nn as nn

def mae_random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, N, D = x.shape 
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


def mask_tokens(x, mask_ratio, mask_token):
    """
    Apply learned mask token to a sequence of embeddings.

    Args:
        x: Tensor [B, N, D] - input token embeddings
        mask_ratio: float - fraction of tokens to mask
        mask_token: nn.Parameter of shape [1, 1, D] - learned mask token

    Returns:
        x_masked: Tensor [B, N, D] - input with masked tokens replaced
        mask: Tensor [B, N] float - 1 for masked, 0 for visible
        mask_indices: Tensor [B, num_masks] - indices of masked tokens
    """
    B, N, D = x.shape
    num_masks = int(N * mask_ratio)

    # generate random permutation of indices for masking
    rand_idx = torch.rand(B, N, device=x.device).argsort(dim=1)
    mask_indices = rand_idx[:, :num_masks]  # first num_masks indices to mask

    # create mask tensor [B, N], 1=masked, 0=visible
    mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(-1, num_masks)
    mask[batch_idx, mask_indices] = 1

    # replace masked positions with learned mask token
    mask_token_expanded = mask_token.expand(B, N, D)  # broadcast
    x_masked = x * (~mask.unsqueeze(-1)) + mask_token_expanded * mask.unsqueeze(-1)

    return x_masked, mask.float(), mask_indices
