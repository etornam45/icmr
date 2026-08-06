"""Proposal generation: multi-threshold grouping, OIC scoring, soft-NMS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Segment:
    start: float
    end: float
    class_id: int
    class_name: str
    confidence: float


def _temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def fuse_pyramid_probs(
    level_logits: list[torch.Tensor],
    target_length: int,
) -> torch.Tensor:
    """Upsample each pyramid level's CAS to stride-1 length and average.

    Args:
        level_logits: list of [B, T_s, C]
        target_length: T at stride 1

    Returns:
        fused probs [B, T, C]
    """
    probs = []
    for logits in level_logits:
        # [B, C, T_s] → interpolate → [B, T, C]
        p = torch.softmax(logits, dim=-1).transpose(1, 2)
        if p.shape[-1] != target_length:
            p = F.interpolate(p, size=target_length, mode="linear", align_corners=False)
        probs.append(p.transpose(1, 2))
    return torch.stack(probs, dim=0).mean(dim=0)


def actionness_from_probs(
    probs: torch.Tensor,
    normal_index: int,
) -> torch.Tensor:
    """a(t) = 1 - p(normal). probs [B, T, C] → [B, T]."""
    return 1.0 - probs[..., normal_index]


def _group_contiguous(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start_idx, end_idx) runs where mask is True."""
    segments: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def oic_score(
    actionness: np.ndarray,
    start_idx: int,
    end_idx: int,
    alpha_margin: float = 0.25,
) -> float:
    """Outer-Inner Contrastive score for a segment on 1D actionness."""
    length = end_idx - start_idx + 1
    if length <= 0:
        return 0.0
    margin = max(1, int(round(alpha_margin * length)))
    inner = actionness[start_idx : end_idx + 1].mean()
    left = actionness[max(0, start_idx - margin) : start_idx]
    right = actionness[end_idx + 1 : min(len(actionness), end_idx + 1 + margin)]
    outer_parts = []
    if left.size:
        outer_parts.append(left.mean())
    if right.size:
        outer_parts.append(right.mean())
    outer = float(np.mean(outer_parts)) if outer_parts else 0.0
    return float(inner - outer)


def multi_threshold_proposals(
    actionness: np.ndarray,
    class_probs: np.ndarray,
    id2label: dict[int, str],
    normal_index: int,
    window_duration: float,
    thresholds: np.ndarray | None = None,
    alpha_margin: float = 0.25,
) -> list[Segment]:
    """Generate OIC-scored segments by sweeping actionness thresholds.

    Args:
        actionness: [T]
        class_probs: [T, C]
        window_duration: seconds spanned by the T timesteps
    """
    t_len = actionness.shape[0]
    if t_len == 0:
        return []
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    def idx_to_time(idx: float) -> float:
        if t_len == 1:
            return 0.0
        return float(idx) / float(t_len - 1) * window_duration

    proposals: list[Segment] = []
    for tau in thresholds:
        for start_i, end_i in _group_contiguous(actionness > float(tau)):
            score = oic_score(actionness, start_i, end_i, alpha_margin=alpha_margin)
            if score <= 0:
                continue
            mean_probs = class_probs[start_i : end_i + 1].mean(axis=0)
            # Prefer non-normal argmax for the segment class.
            mean_probs = mean_probs.copy()
            mean_probs[normal_index] = -1.0
            class_id = int(mean_probs.argmax())
            class_name = id2label.get(class_id, str(class_id))
            if class_name.lower() in {"normal", "normal_videos", "none", "n/a"}:
                continue
            proposals.append(
                Segment(
                    start=idx_to_time(start_i),
                    end=idx_to_time(end_i),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(score),
                )
            )
    return proposals


def soft_nms(
    segments: list[Segment],
    sigma: float = 0.5,
    score_floor: float = 0.05,
    max_keep: int = 100,
) -> list[Segment]:
    """Gaussian soft-NMS on 1D temporal segments."""
    if not segments:
        return []
    remaining = sorted(segments, key=lambda s: s.confidence, reverse=True)
    kept: list[Segment] = []
    while remaining and len(kept) < max_keep:
        best = remaining.pop(0)
        if best.confidence < score_floor:
            break
        kept.append(best)
        updated: list[Segment] = []
        for cand in remaining:
            iou = _temporal_iou(best.start, best.end, cand.start, cand.end)
            new_score = cand.confidence * float(np.exp(-(iou ** 2) / sigma))
            if new_score >= score_floor:
                updated.append(
                    Segment(
                        start=cand.start,
                        end=cand.end,
                        class_id=cand.class_id,
                        class_name=cand.class_name,
                        confidence=new_score,
                    )
                )
        remaining = sorted(updated, key=lambda s: s.confidence, reverse=True)
    return kept


def merge_same_class(
    segments: list[Segment],
    gap_sec: float = 1.0,
) -> list[Segment]:
    """Absorb near-duplicate same-class segments with small temporal gaps."""
    if not segments:
        return []
    by_class: dict[int, list[Segment]] = {}
    for seg in segments:
        by_class.setdefault(seg.class_id, []).append(seg)

    merged: list[Segment] = []
    for class_id, group in by_class.items():
        group = sorted(group, key=lambda s: s.start)
        cur = group[0]
        for nxt in group[1:]:
            if nxt.start <= cur.end + gap_sec:
                cur = Segment(
                    start=cur.start,
                    end=max(cur.end, nxt.end),
                    class_id=class_id,
                    class_name=cur.class_name,
                    confidence=max(cur.confidence, nxt.confidence),
                )
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
    return sorted(merged, key=lambda s: s.confidence, reverse=True)


def decode_segments(
    level_logits: list[torch.Tensor],
    id2label: dict[int, str],
    normal_index: int,
    window_duration: float,
    nms_sigma: float = 0.5,
    nms_floor: float = 0.05,
    deploy_threshold: float = 0.0,
    alpha_margin: float = 0.25,
    merge_gap_sec: float = 1.0,
) -> list[Segment]:
    """Full decode path for a single video (batch size 1 expected).

    Args:
        level_logits: list of [1, T_s, C] or [T_s, C]
    """
    levels = []
    for logits in level_logits:
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        levels.append(logits)
    target_t = levels[0].shape[1]
    probs = fuse_pyramid_probs(levels, target_t)[0]  # [T, C]
    action = actionness_from_probs(probs.unsqueeze(0), normal_index)[0]

    action_np = action.detach().float().cpu().numpy()
    probs_np = probs.detach().float().cpu().numpy()

    proposals = multi_threshold_proposals(
        action_np,
        probs_np,
        id2label=id2label,
        normal_index=normal_index,
        window_duration=window_duration,
        alpha_margin=alpha_margin,
    )
    suppressed = soft_nms(proposals, sigma=nms_sigma, score_floor=nms_floor)
    merged = merge_same_class(suppressed, gap_sec=merge_gap_sec)
    if deploy_threshold > 0:
        merged = [s for s in merged if s.confidence >= deploy_threshold]
    return merged
