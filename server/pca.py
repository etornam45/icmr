"""DINO patch-token PCA → RGB heatmap overlay."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def patch_pca_rgb(
    patch_tokens: torch.Tensor,
    grid_size: int | None = None,
) -> np.ndarray:
    """
    Project patch tokens to 3 PCA components and map to [0, 1] RGB.

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

    mean = tokens.mean(axis=0, keepdims=True)
    centered = tokens - mean
    # Economy SVD on centered features; first 3 right-singular vectors ≈ PCA axes.
    # For tall matrices (N < D) use thin SVD on Gram or sklearn-style: SVD of centered.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]  # [3, D]
    projected = centered @ components.T  # [N, 3]

    # Robust per-channel stretch to [0, 1]
    rgb = np.zeros_like(projected, dtype=np.float32)
    for channel in range(3):
        values = projected[:, channel]
        lo, hi = np.percentile(values, 2), np.percentile(values, 98)
        if hi <= lo:
            rgb[:, channel] = 0.5
        else:
            rgb[:, channel] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    grid = (rgb.reshape(grid_size, grid_size, 3) * 255.0).astype(np.uint8)
    return grid


def upsample_heatmap(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest/bilinear upsample HxWx3 heatmap to target size (OpenCV)."""
    import cv2

    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)


def blend_pca_overlay(
    frame_bgr: np.ndarray,
    patch_tokens: torch.Tensor,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend PCA RGB heatmap onto a BGR frame."""
    import cv2

    h, w = frame_bgr.shape[:2]
    heatmap = patch_pca_rgb(patch_tokens)
    heat_bgr = cv2.cvtColor(
        upsample_heatmap(heatmap, w, h), cv2.COLOR_RGB2BGR
    )
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat_bgr, alpha, 0)
