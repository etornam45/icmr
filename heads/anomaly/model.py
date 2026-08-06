"""DINOv3 weakly-supervised temporal anomaly localizer (WTAL)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dinov3.checkpoints.load import (
    ensure_backbone_checkpoint,
    load_checkpoint,
    validate_checkpoint_file,
)
from dinov3.models import vit_small
from heads.anomaly.decode import (
    actionness_from_probs,
    decode_segments,
    fuse_pyramid_probs,
)
from heads.anomaly.heads import BoundaryHead, ClassificationHead
from heads.anomaly.losses import SVDDRegularizer
from heads.anomaly.pyramid import FeaturePyramid1D
from heads.anomaly.temporal import TemporalAggregator

ARCHITECTURE = "temporal_wtal_v1"
BACKBONE_WEIGHTS = "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_CHECKPOINT_DIR = "dinov3/checkpoints/model/anomaly_vau"
VISION_DIM = 384
IMG_SIZE = 224
DEFAULT_NUM_FRAMES = 16
DEFAULT_STRIDES = (1, 2, 4, 8)


class DINOv3AnomalyLocalizer(nn.Module):
    """Frozen DINOv3 CLS → temporal aggregator → pyramid → CAS head.

    Boundary regression head is present but inactive (``boundary.enabled=False``).
    """

    def __init__(
        self,
        vision_model: nn.Module,
        num_classes: int,
        vision_dim: int = VISION_DIM,
        hidden_dim: int = 256,
        strides: tuple[int, ...] = DEFAULT_STRIDES,
        normal_index: int = 0,
    ):
        super().__init__()
        self.architecture = ARCHITECTURE
        self.vision_model = vision_model
        self.num_classes = num_classes
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.strides = tuple(strides)
        self.normal_index = normal_index

        self.aggregator = TemporalAggregator(dim=vision_dim)
        self.pyramid = FeaturePyramid1D(dim=vision_dim, strides=self.strides)
        self.cls_head = ClassificationHead(
            dim=vision_dim, num_classes=num_classes, hidden_dim=hidden_dim
        )
        self.boundary = BoundaryHead(dim=vision_dim, hidden_dim=hidden_dim)
        self.boundary.enabled = False

        self.svdd = SVDDRegularizer(dim=vision_dim)
        self.svdd.attach(self)

    def encode_cls(self, videos: torch.Tensor) -> torch.Tensor:
        """videos [B,T,3,H,W] → CLS [B,T,D] (backbone frozen)."""
        if videos.dim() == 4:
            videos = videos.unsqueeze(1)
        if videos.dim() != 5:
            raise ValueError(
                f"Expected video tensor [B, T, 3, H, W], got shape {tuple(videos.shape)}"
            )
        batch_size, num_frames, channels, height, width = videos.shape
        flat = videos.reshape(batch_size * num_frames, channels, height, width)
        with torch.no_grad():
            features = self.vision_model(flat, masks=None, is_training=True)
            cls_tokens = features["x_norm_clstoken"]
        return cls_tokens.view(batch_size, num_frames, -1)

    def forward_from_cls(self, cls_tokens: torch.Tensor) -> dict[str, object]:
        """
        Args:
            cls_tokens: [B, T, D]

        Returns:
            embeddings: [B, T, D] post-aggregator
            level_logits: list of [B, T_s, C]
            cas_logits: [B, T, C] stride-1
            fused_probs: [B, T, C]
            actionness: [B, T]
        """
        if cls_tokens.dim() == 2:
            cls_tokens = cls_tokens.unsqueeze(0)
        embeddings = self.aggregator(cls_tokens)
        levels = self.pyramid(embeddings)
        level_logits = [self.cls_head(level) for level in levels]
        cas_logits = level_logits[0]
        fused_probs = fuse_pyramid_probs(level_logits, target_length=cas_logits.shape[1])
        actionness = actionness_from_probs(fused_probs, self.normal_index)
        return {
            "embeddings": embeddings,
            "level_logits": level_logits,
            "cas_logits": cas_logits,
            "fused_probs": fused_probs,
            "actionness": actionness,
        }

    def forward(self, videos: torch.Tensor) -> dict[str, object]:
        cls_tokens = self.encode_cls(videos)
        return self.forward_from_cls(cls_tokens)

    def video_logits(self, cas_logits: torch.Tensor, k_ratio: float = 1.0 / 8.0):
        """Top-k pooled video-level logits [B, C] for classification metrics."""
        import math

        _batch, num_steps, _c = cas_logits.shape
        k = max(1, math.ceil(num_steps * k_ratio))
        topk_vals, _ = torch.topk(cas_logits.transpose(1, 2), k=k, dim=-1)
        return topk_vals.mean(dim=-1)


# Back-compat alias used by older imports / type hints.
DINOv3AnomalyClassifier = DINOv3AnomalyLocalizer


def localizer_trainable_params(model: DINOv3AnomalyLocalizer):
    params = (
        list(model.aggregator.parameters())
        + list(model.pyramid.parameters())
        + list(model.cls_head.parameters())
    )
    if model.boundary.enabled:
        params += list(model.boundary.parameters())
    return params


# Alias for older train script name.
classifier_trainable_params = localizer_trainable_params


def build_anomaly_model(
    device: torch.device,
    num_classes: int,
    backbone_weights: str = BACKBONE_WEIGHTS,
    hidden_dim: int = 256,
    vision_model: nn.Module | None = None,
    normal_index: int = 0,
    strides: tuple[int, ...] = DEFAULT_STRIDES,
) -> DINOv3AnomalyLocalizer:
    if vision_model is None:
        vision_model = vit_small(
            patch_size=16,
            n_storage_tokens=4,
            layerscale_init=1e-5,
            mask_k_bias=True,
        )
        if Path(backbone_weights).exists() and not validate_checkpoint_file(
            backbone_weights, expected_sha256=None
        ):
            print(
                f"Warning: checkpoint at {backbone_weights} looks corrupt, re-downloading"
            )
            Path(backbone_weights).unlink(missing_ok=True)

        backbone_weights = ensure_backbone_checkpoint(backbone_weights)
        load_checkpoint(vision_model, backbone_weights)
        vision_model.to(device)
        vision_model.eval()
        for param in vision_model.parameters():
            param.requires_grad = False

    model = DINOv3AnomalyLocalizer(
        vision_model=vision_model,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        strides=strides,
        normal_index=normal_index,
    ).to(device)
    return model


def save_anomaly_checkpoint(
    model: DINOv3AnomalyLocalizer,
    checkpoint_dir: str | Path,
    label2id: dict[str, int],
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "aggregator": model.aggregator.state_dict(),
            "pyramid": model.pyramid.state_dict(),
            "cls_head": model.cls_head.state_dict(),
            "boundary": model.boundary.state_dict(),
            "svdd_center": model.svdd_center.detach().cpu(),
            "svdd_initialized": int(model.svdd_initialized.item()),
            "num_classes": model.num_classes,
            "hidden_dim": model.hidden_dim,
            "strides": list(model.strides),
            "normal_index": model.normal_index,
            "label2id": label2id,
        },
        path / "classifier.pt",
    )
    from heads.vqa.vau_dataset import save_label_maps

    save_label_maps(label2id, path / "label2id.json")


def load_anomaly_checkpoint(
    model: DINOv3AnomalyLocalizer,
    checkpoint_dir: str | Path,
    device: torch.device,
) -> dict[str, int]:
    path = Path(checkpoint_dir)
    ckpt_path = path / "classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No classifier checkpoint at {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = state.get("architecture")
    if arch is not None and arch != ARCHITECTURE:
        raise RuntimeError(
            f"Checkpoint architecture={arch!r} is incompatible with "
            f"{ARCHITECTURE!r}. Retrain with heads.anomaly.train."
        )
    # Legacy mean-pool checkpoints have "norm"/"head" and no architecture key.
    if "aggregator" not in state:
        raise RuntimeError(
            "Legacy mean-pool anomaly checkpoint detected. "
            f"Retrain for {ARCHITECTURE}."
        )

    num_classes = int(state["num_classes"])
    if num_classes != model.num_classes:
        raise RuntimeError(
            f"Checkpoint num_classes={num_classes} != model "
            f"num_classes={model.num_classes}"
        )
    model.aggregator.load_state_dict(state["aggregator"])
    model.pyramid.load_state_dict(state["pyramid"])
    model.cls_head.load_state_dict(state["cls_head"])
    if "boundary" in state:
        model.boundary.load_state_dict(state["boundary"])
    if "svdd_center" in state:
        model.svdd_center.copy_(state["svdd_center"].to(device))
        model.svdd_initialized.fill_(int(state.get("svdd_initialized", 0)))
    if "normal_index" in state:
        model.normal_index = int(state["normal_index"])
    label2id = {str(k): int(v) for k, v in state["label2id"].items()}
    return label2id


def localize_from_cls(
    model: DINOv3AnomalyLocalizer,
    cls_tokens: torch.Tensor,
    id2label: dict[int, str],
    window_duration: float,
    nms_sigma: float = 0.5,
    nms_floor: float = 0.05,
    deploy_threshold: float = 0.0,
) -> dict:
    """Run localization from precomputed CLS tokens (shared-backbone path)."""
    out = model.forward_from_cls(cls_tokens)
    level_logits = out["level_logits"]
    # decode expects batch dim; take first item levels as list of [1,T,C]
    segments = decode_segments(
        [logits[:1] for logits in level_logits],
        id2label=id2label,
        normal_index=model.normal_index,
        window_duration=window_duration,
        nms_sigma=nms_sigma,
        nms_floor=nms_floor,
        deploy_threshold=deploy_threshold,
    )
    video_logits = model.video_logits(out["cas_logits"])
    probs = torch.softmax(video_logits[0], dim=-1)
    top_prob, top_idx = probs.max(dim=-1)
    prediction = id2label[int(top_idx.item())]
    actionness = out["actionness"][0].detach().float().cpu()
    svdd_dist = model.svdd.distances(out["embeddings"])[0].detach().float().cpu()
    return {
        "prediction": prediction,
        "score": float(top_prob.item()),
        "segments": segments,
        "actionness": actionness,
        "svdd_distance": float(svdd_dist.mean().item()),
        "top_k": [
            {"class": id2label[int(i)], "probability": float(probs[int(i)].item())}
            for i in torch.topk(probs, k=min(5, probs.numel())).indices.tolist()
        ],
    }


if __name__ == "__main__":
    device = torch.device("cpu")
    # Tiny smoke without loading real backbone weights.
    class _FakeVision(nn.Module):
        def forward(self, x, masks=None, is_training=True):
            b = x.shape[0]
            return {
                "x_norm_clstoken": torch.randn(b, VISION_DIM),
            }

    model = DINOv3AnomalyLocalizer(
        vision_model=_FakeVision(),
        num_classes=12,
        normal_index=7,
    )
    videos = torch.randn(2, DEFAULT_NUM_FRAMES, 3, 32, 32)
    # Bypass encode (fake vision ignores spatial size) via forward_from_cls
    cls = torch.randn(2, DEFAULT_NUM_FRAMES, VISION_DIM)
    out = model.forward_from_cls(cls)
    assert out["cas_logits"].shape == (2, DEFAULT_NUM_FRAMES, 12)
    assert len(out["level_logits"]) == 4
    print(
        "smoke ok:",
        f"cas={tuple(out['cas_logits'].shape)}",
        f"levels={[tuple(x.shape) for x in out['level_logits']]}",
        f"actionness={tuple(out['actionness'].shape)}",
    )
