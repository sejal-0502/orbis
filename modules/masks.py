import torch

def random_mask_tokens(x, mask_ratio):
    """
    x: [B, N, D] patch tokens
    mask_ratio: fraction of patches to mask
    Returns:
        unmasked_x: [B, N_kept, D] tokens kept
        mask: [B, N] boolean mask (True = masked)
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    # Random permutation per batch
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]

    # Gather unmasked tokens
    batch_idx = torch.arange(B)[:, None]
    unmasked_x = x[batch_idx, ids_keep]

    # Create mask boolean tensor
    masked_x = torch.ones(B, N, device=x.device, dtype=torch.bool)
    masked_x[batch_idx, ids_keep] = False

    return unmasked_x, masked_x, ids_restore, len_keep