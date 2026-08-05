"""DINOv3 + Qwen2.5 Flamingo-style video VQA (gated cross-attention)."""

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
from heads.vqa.gated_xattn import (
    VisualFeatureHolder,
    build_gated_blocks,
    count_xattn_slots,
    wrap_decoder_layers,
)
from heads.vqa.llm_loader import (
    LLM_MODEL_NAME,
    get_decoder_layers,
    load_llm,
    load_llm_tokenizer,
)
from heads.vqa.resampler import PerceiverResampler

BACKBONE_WEIGHTS = "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_CHECKPOINT_DIR = "dinov3/checkpoints/model/vqa_cuva_qwen"
VISION_DIM = 384
LLM_DIM = 1536
IMG_SIZE = 224
PATCH_SIZE = 16
DEFAULT_NUM_FRAMES = 16
DEFAULT_NUM_LATENTS = 64
DEFAULT_RESAMPLER_DEPTH = 3
DEFAULT_XATTN_EVERY_N = 4
DEFAULT_NUM_HEADS = 12  # 1536 / 12 = 128
ARCHITECTURE = "video_xattn_v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(videos: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std to video/image tensors in [0, 1]."""
    if videos.dim() == 5:
        shape = (1, 1, 3, 1, 1)
    elif videos.dim() == 4:
        shape = (1, 3, 1, 1)
    elif videos.dim() == 3:
        shape = (3, 1, 1)
    else:
        raise ValueError(f"Expected 3D/4D/5D tensor, got shape {tuple(videos.shape)}")
    mean = videos.new_tensor(IMAGENET_MEAN).view(*shape)
    std = videos.new_tensor(IMAGENET_STD).view(*shape)
    return (videos - mean) / std


class DINOv3QwenXAttn(nn.Module):
    """Video VQA: DINOv3 patches → Perceiver → gated cross-attn into Qwen."""

    def __init__(
        self,
        vision_model: nn.Module,
        llm: nn.Module,
        resampler: PerceiverResampler,
        xattn_blocks: nn.ModuleList,
        holder: VisualFeatureHolder,
        num_frames: int = DEFAULT_NUM_FRAMES,
        xattn_every_n: int = DEFAULT_XATTN_EVERY_N,
        vision_dim: int = VISION_DIM,
        llm_dim: int = LLM_DIM,
        patch_size: int = PATCH_SIZE,
        img_size: int = IMG_SIZE,
        xattn_layer_indices: list[int] | None = None,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.llm = llm
        self.resampler = resampler
        self.xattn_blocks = xattn_blocks
        self.holder = holder
        self.num_frames = num_frames
        self.xattn_every_n = xattn_every_n
        self.num_visual_tokens = resampler.num_latents
        self.llm_dim = llm_dim
        self.vision_dim = vision_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        self.xattn_layer_indices = list(xattn_layer_indices or [])

    def encode_visual(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encode videos into fixed Perceiver latents.

        Args:
            videos: [B, T, 3, H, W] or [B, 3, H, W] in [0, 1]

        Returns:
            visual_embeds: [B, num_latents, llm_dim]
        """
        if videos.dim() == 4:
            videos = videos.unsqueeze(1)
        if videos.dim() != 5:
            raise ValueError(
                f"Expected video tensor [B, T, 3, H, W], got shape {tuple(videos.shape)}"
            )

        batch_size, num_frames, channels, height, width = videos.shape
        if height != width:
            raise ValueError(f"Expected square frames, got HxW={height}x{width}")
        grid = height // self.patch_size
        if grid * self.patch_size != height:
            raise ValueError(
                f"Frame size {height} must be divisible by patch size {self.patch_size}"
            )

        videos = imagenet_normalize(videos)
        flat = videos.reshape(batch_size * num_frames, channels, height, width)

        with torch.no_grad():
            features = self.vision_model(flat, masks=None, is_training=True)
            cls_tokens = features["x_norm_clstoken"]
            patch_tokens = features["x_norm_patchtokens"]

        _, num_patches, dim = patch_tokens.shape
        if num_patches != grid * grid:
            raise ValueError(f"Expected {grid * grid} patch tokens, got {num_patches}")

        patches = patch_tokens.view(batch_size, num_frames, num_patches, dim)
        cls = cls_tokens.view(batch_size, num_frames, dim)
        return self.resampler(patches, cls_tokens=cls)

    def _run_llm(
        self,
        visual_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ):
        self.holder.set(visual_embeds)
        try:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        finally:
            self.holder.clear()

    def forward(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        visual_embeds = self.encode_visual(videos)
        return self._run_llm(visual_embeds, input_ids, attention_mask, labels)

    @torch.no_grad()
    def generate(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ) -> torch.Tensor:
        visual_embeds = self.encode_visual(videos)
        self.holder.set(visual_embeds)
        was_gc = getattr(self.llm, "is_gradient_checkpointing", False)
        if was_gc and hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()
        try:
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                use_cache=True,
            )
        finally:
            self.holder.clear()
            if was_gc and hasattr(self.llm, "gradient_checkpointing_enable"):
                self.llm.gradient_checkpointing_enable()


# Backward-compatible alias for callers that still import the old name.
DINOv3MiniCPMHybrid = DINOv3QwenXAttn


def decode_generated_answer(
    tokenizer,
    gen_ids: torch.Tensor,
    prompt_len: int,
    num_visual_tokens: int = 0,
) -> str:
    """Decode newly generated tokens. Visual tokens are not in the id sequence."""
    del num_visual_tokens  # unused; kept for call-site compatibility
    new_tokens = gen_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_hybrid_model(
    device: torch.device,
    llm_model_name: str = LLM_MODEL_NAME,
    backbone_weights: str = BACKBONE_WEIGHTS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_latents: int = DEFAULT_NUM_LATENTS,
    resampler_depth: int = DEFAULT_RESAMPLER_DEPTH,
    xattn_every_n: int = DEFAULT_XATTN_EVERY_N,
    num_heads: int = DEFAULT_NUM_HEADS,
    lora_r: int = 16,
    lora_alpha: int = 32,
    adapter_path: str | None = None,
    vision_model: nn.Module | None = None,
    apply_lora: bool = True,
):
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

    llm = load_llm(
        device,
        model_name=llm_model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        adapter_path=adapter_path,
        apply_lora=apply_lora,
    )

    num_layers = len(get_decoder_layers(llm))
    num_blocks = count_xattn_slots(num_layers, xattn_every_n)
    holder = VisualFeatureHolder()
    xattn_blocks = build_gated_blocks(num_blocks, LLM_DIM, num_heads)
    xattn_blocks.to(device)
    indices = wrap_decoder_layers(llm, xattn_blocks, holder, every_n=xattn_every_n)

    grid_size = IMG_SIZE // PATCH_SIZE
    resampler = PerceiverResampler(
        vision_dim=VISION_DIM,
        llm_dim=LLM_DIM,
        num_latents=num_latents,
        depth=resampler_depth,
        num_heads=num_heads,
        max_frames=max(num_frames, 32),
        grid_size=grid_size,
        include_cls=True,
    ).to(device)

    model = DINOv3QwenXAttn(
        vision_model=vision_model,
        llm=llm,
        resampler=resampler,
        xattn_blocks=xattn_blocks,
        holder=holder,
        num_frames=num_frames,
        xattn_every_n=xattn_every_n,
        xattn_layer_indices=indices,
    ).to(device)

    if hasattr(model.llm, "enable_input_require_grads"):
        model.llm.enable_input_require_grads()

    tokenizer = load_llm_tokenizer(llm_model_name)
    return model, tokenizer


def visual_params(model: DINOv3QwenXAttn):
    """Trainable visual pathway: resampler + gated cross-attention blocks."""
    return list(model.resampler.parameters()) + list(model.xattn_blocks.parameters())


def vision_adapter_params(model: DINOv3QwenXAttn):
    """Alias kept for train.py compatibility."""
    return visual_params(model)


def adapter_trainable_params(model: DINOv3QwenXAttn):
    return [p for p in model.llm.parameters() if p.requires_grad]


def save_hybrid_checkpoint(model: DINOv3QwenXAttn, checkpoint_dir: str) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(path / "adapter")
    torch.save(
        {
            "resampler": model.resampler.state_dict(),
            "xattn_blocks": model.xattn_blocks.state_dict(),
            "num_frames": model.num_frames,
            "num_latents": model.num_visual_tokens,
            "num_visual_tokens": model.num_visual_tokens,
            "xattn_every_n": model.xattn_every_n,
            "xattn_layer_indices": model.xattn_layer_indices,
            "architecture": ARCHITECTURE,
        },
        path / "visual_head.pt",
    )


def load_hybrid_checkpoint(
    model: DINOv3QwenXAttn,
    checkpoint_dir: str,
    device: torch.device,
    trainable_adapter: bool = False,
) -> None:
    from peft import PeftModel

    path = Path(checkpoint_dir)
    adapter_dir = path / "adapter"
    visual_path = path / "visual_head.pt"
    # Legacy prefix-concat adapters are not compatible.
    legacy_path = path / "vision_adapter.pt"

    if adapter_dir.exists():
        if isinstance(model.llm, PeftModel):
            base_model = model.llm.get_base_model()
        else:
            base_model = model.llm
        # Decoder layers are already xattn-wrapped on the base; do not wrap again.
        model.llm = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=trainable_adapter,
        ).to(device)

    if visual_path.exists():
        state = torch.load(visual_path, map_location=device, weights_only=True)
        arch = state.get("architecture")
        if arch != ARCHITECTURE:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_dir} has architecture={arch!r}, "
                f"expected {ARCHITECTURE!r}. Older video_spatial_* / MiniCPM "
                "adapters are not compatible; retrain with video_xattn_v1."
            )
        ckpt_frames = state.get("num_frames")
        if ckpt_frames is not None and int(ckpt_frames) != model.num_frames:
            print(
                f"Warning: checkpoint num_frames={ckpt_frames} != model "
                f"num_frames={model.num_frames}; using model setting."
            )
        ckpt_latents = state.get("num_latents", state.get("num_visual_tokens"))
        if ckpt_latents is not None and int(ckpt_latents) != model.num_visual_tokens:
            raise RuntimeError(
                f"Checkpoint num_latents={ckpt_latents} != model "
                f"num_latents={model.num_visual_tokens}."
            )
        model.resampler.load_state_dict(state["resampler"])
        model.xattn_blocks.load_state_dict(state["xattn_blocks"])
        return

    if legacy_path.exists():
        raise RuntimeError(
            f"Found legacy vision_adapter.pt at {checkpoint_dir} "
            f"(expected {ARCHITECTURE} visual_head.pt). Retrain required."
        )
    raise FileNotFoundError(f"No visual_head.pt found under {checkpoint_dir}")


if __name__ == "__main__":
    from dinov3.utils.device import get_device
    from heads.vqa.llm_loader import tokenize_chat_pair

    device = get_device()
    num_frames = DEFAULT_NUM_FRAMES
    model, tokenizer = build_hybrid_model(device, num_frames=num_frames)

    videos = torch.randn(1, num_frames, 3, IMG_SIZE, IMG_SIZE, device=device)
    questions = ["Describe what happens in the video."]
    answers = ["A person approaches another and starts a fight near a doorway."]

    input_ids = []
    labels = []
    for question, answer in zip(questions, answers):
        ids, lbls = tokenize_chat_pair(tokenizer, question, answer, max_length=128)
        input_ids.append(ids)
        labels.append(lbls)

    max_len = max(len(row) for row in input_ids)
    pad_id = tokenizer.pad_token_id
    input_ids_t = torch.full((1, max_len), pad_id, dtype=torch.long, device=device)
    labels_t = torch.full((1, max_len), -100, dtype=torch.long, device=device)
    attn = torch.zeros((1, max_len), dtype=torch.long, device=device)
    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        input_ids_t[i, : len(ids)] = torch.tensor(ids, device=device)
        labels_t[i, : len(lbls)] = torch.tensor(lbls, device=device)
        attn[i, : len(ids)] = 1

    visual = model.encode_visual(videos)
    print("visual_embeds:", tuple(visual.shape))
    print("num_visual_tokens:", model.num_visual_tokens)
    print("xattn layers:", model.xattn_layer_indices)
    out = model(videos, input_ids_t, attn, labels=labels_t)
    print("loss:", float(out.loss.detach().cpu()))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.1f}M")
