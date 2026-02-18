# JEPA Implementation (Predictive Self-Supervised Learning)

Implementation of **JEPA** using ViT encoders and a small transformer predictor. It follows a masked-image predictive setup to learn semantically rich patch embeddings.

Model settings can be found under : **`configs/stage1_jepa.yaml`**.

---

## Overview

Approach :

- A **context encoder** sees a masked version of the input image.
- A **target encoder** sees the full image.
- A **predictor** predicts the context encoder’s masked embeddings conditioned on their positional embeddings.
- The **loss** encourages the predictor output to match the corresponding target encoder embeddings.

Goal : This setup encourages the context encoder to produce **semantically meaningful representations**.

---

## Architecture

### Context Encoder (ViT Base)
- Input: Masked image `[B, 3, H, W]`
- Output: Patch embeddings `[B, N_patches, D]` (e.g., `[B, 256, 768]`)
- **Gradient:** Backpropagated during training
- **Masking:** Random subset of patches is replaced with a learned mask token

### Target Encoder (ViT Base)
- Input: Full image `[B, 3, H, W]`
- Output: Patch embeddings `[B, N_patches, D]` (e.g., `[B, 256, 768]`)
- **Gradient:** Stop-gradient
- **Update:** EMA from context encoder weights
- **Purpose:** Provides a target for the predictor

### Predictor
- Input: Masked patch embeddings from context encoder `[B, N_masks, D_pred]`
- Conditioning: Positional embeddings for the masked patches `[B, N_masks, D_pred]`
- Output: Predicted embeddings `[B, N_masks, D_pred]` (e.g., `[B, N_masks, 384]`)
- **Gradient:** Backpropagated during training

### Loss
- **Projection:** Predicted tokens and target tokens are projected to the same embedding dimension `D = 384`
```python
loss = F.l1_loss(pred_tokens, target_tokens)

