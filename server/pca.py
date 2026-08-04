"""DINO patch-token PCA → RGB heatmap (pure, no video blend)."""

from __future__ import annotations

import numpy as np
import torch


def _normalize_channel(values: np.ndarray) -> np.ndarray:
    """High-contrast stretch with mild mid-tone boost for patch structure."""
    values = values.astype(np.float32)
    lo, hi = np.percentile(values, 1), np.percentile(values, 99)
    if hi <= lo:
        return np.full(values.shape, 0.5, dtype=np.float32)
    stretched = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.power(stretched, 0.75).astype(np.float32)


@torch.no_grad()
def patch_pca_rgb(
    patch_tokens: torch.Tensor,
    grid_size: int | None = None,
) -> np.ndarray:
    """
    Map patch embeddings to an RGB heatmap via the top-3 PCA components.

    Tokens are L2-normalized before PCA so directional structure reads clearly.

    Args:
        patch_tokens: [N, D] or [1, N, D]

    Returns:
        rgb: uint8 array [H_patches, W_patches, 3]
    """
    if patch_tokens.dim() == 3:
        patch_tokens = patch_tokens[0]
    tokens = patch_tokens.detach().float().cpu().numpy()
    n_patches, _dim = tokens.shape
    if grid_size is None:
        grid_size = round(n_patches**0.5)
    if grid_size * grid_size != n_patches:
        raise ValueError(
            f"Expected square patch grid, got N={n_patches} (grid_size={grid_size})"
        )

    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    tokens = tokens / np.clip(norms, 1e-8, None)

    mean = tokens.mean(axis=0, keepdims=True)
    centered = tokens - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:3].T  # [N, 3]

    # Local contrast per component so neighboring patches separate more.
    rgb = np.zeros((n_patches, 3), dtype=np.float32)
    for channel in range(3):
        spatial = projected[:, channel].reshape(grid_size, grid_size)
        spatial = spatial - spatial.mean()
        rgb[:, channel] = _normalize_channel(spatial.reshape(-1))

    return (rgb.reshape(grid_size, grid_size, 3) * 255.0).astype(np.uint8)


def upsample_heatmap(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    """Upsample patch grid with nearest-neighbor to keep crisp patch boundaries."""
    import cv2

    h0, w0 = heatmap.shape[:2]
    mid_w = max(w0 * 8, min(width, w0 * 16))
    mid_h = max(h0 * 8, min(height, h0 * 16))
    crisp = cv2.resize(heatmap, (mid_w, mid_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(crisp, (width, height), interpolation=cv2.INTER_LINEAR)


def blend_pca_overlay(
    frame_bgr: np.ndarray,
    patch_tokens: torch.Tensor,
    alpha: float = 0.55,
) -> np.ndarray:
    """Render pure PCA RGB heatmap (no video blend), sized to the frame."""
    import cv2

    del alpha  # kept for call-site compatibility; PCA mode is not blended
    h, w = frame_bgr.shape[:2]
    heatmap = patch_pca_rgb(patch_tokens)
    return cv2.cvtColor(upsample_heatmap(heatmap, w, h), cv2.COLOR_RGB2BGR)
