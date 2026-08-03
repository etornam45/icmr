from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from dinov3.checkpoints.load import (
    ensure_backbone_checkpoint,
    load_checkpoint,
    validate_checkpoint_file,
)
from dinov3.models import vit_small
from heads.vqa.minicpm_loader import (
    MINICPM_MODEL_NAME,
    load_minicpm_llm,
    load_minicpm_tokenizer,
)

BACKBONE_WEIGHTS = "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_CHECKPOINT_DIR = "dinov3/checkpoints/model/vqa_cuva_minicpm"
VISION_DIM = 384
LLM_DIM = 1536
IMG_SIZE = 224
PATCH_SIZE = 16
DEFAULT_NUM_FRAMES = 16 * 2


def build_1d_sincos_pos_embed(num_positions: int, dim: int) -> torch.Tensor:
    """Sinusoidal 1D temporal position encoding: (1, T, dim)."""
    assert dim % 2 == 0, "pos embed dim must be even"
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    omega = 1.0 / (10000**omega)
    angles = position * omega.unsqueeze(0)
    pos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return pos.unsqueeze(0)


class DINOv3MiniCPMHybrid(nn.Module):
    """Video VQA: one MiniCPM visual token per frame from DINOv3 CLS."""

    def __init__(
        self,
        vision_model: nn.Module,
        llm: nn.Module,
        num_frames: int = DEFAULT_NUM_FRAMES,
        vision_dim: int = VISION_DIM,
        llm_dim: int = LLM_DIM,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.llm = llm
        self.num_frames = num_frames
        self.num_visual_tokens = num_frames
        self.llm_dim = llm_dim

        self.vision_projection = nn.Linear(vision_dim, llm_dim)
        self.projection_norm = nn.LayerNorm(llm_dim)
        self.register_buffer(
            "_pos", build_1d_sincos_pos_embed(num_frames, llm_dim)
        )

    def _llm_dtype(self) -> torch.dtype:
        return next(self.llm.parameters()).dtype

    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def encode_visual(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encode videos into ordered CLS visual prefix tokens.

        Args:
            videos: [B, T, 3, H, W] or [B, 3, H, W] (single-frame fallback)

        Returns:
            visual_embeds: [B, T, llm_dim]
        """
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

        cls_tokens = cls_tokens.view(batch_size, num_frames, -1)
        projected = self.projection_norm(self.vision_projection(cls_tokens))

        pos = self._pos[:, :num_frames, :].to(dtype=projected.dtype, device=projected.device)
        if num_frames > self._pos.shape[1]:
            # Extend temporal PE on the fly if more frames than registered buffer.
            pos = build_1d_sincos_pos_embed(num_frames, self.llm_dim).to(
                device=projected.device, dtype=projected.dtype
            )
        return projected + pos

    def _merge_visual_and_text(
        self,
        visual_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ):
        text_embeds = self._get_text_embeddings(input_ids)
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        batch_size, visual_len, _ = visual_embeds.shape
        visual_mask = torch.ones(
            batch_size,
            visual_len,
            device=inputs_embeds.device,
            dtype=attention_mask.dtype if attention_mask is not None else torch.long,
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape, device=input_ids.device, dtype=torch.long
            )
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full(
                (batch_size, visual_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        visual_embeds = self.encode_visual(videos)
        inputs_embeds, attention_mask, labels = self._merge_visual_and_text(
            visual_embeds, input_ids, attention_mask, labels
        )
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ) -> torch.Tensor:
        visual_embeds = self.encode_visual(videos)
        inputs_embeds, attention_mask, _ = self._merge_visual_and_text(
            visual_embeds, input_ids, attention_mask, labels=None
        )
        visual_len = visual_embeds.shape[1]
        visual_placeholders = torch.zeros(
            input_ids.shape[0],
            visual_len,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        full_input_ids = torch.cat([visual_placeholders, input_ids], dim=1)
        was_gc = getattr(self.llm, "is_gradient_checkpointing", False)
        if was_gc and hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()
        try:
            return self.llm.generate(
                input_ids=full_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                use_cache=True,
            )
        finally:
            if was_gc and hasattr(self.llm, "gradient_checkpointing_enable"):
                self.llm.gradient_checkpointing_enable()


def decode_generated_answer(
    tokenizer,
    gen_ids: torch.Tensor,
    prompt_len: int,
    num_visual_tokens: int = DEFAULT_NUM_FRAMES,
) -> str:
    start = num_visual_tokens + prompt_len
    new_tokens = gen_ids[0, start:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_hybrid_model(
    device: torch.device,
    llm_model_name: str = MINICPM_MODEL_NAME,
    backbone_weights: str = BACKBONE_WEIGHTS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    lora_r: int = 16,
    lora_alpha: int = 32,
    adapter_path: Optional[str] = None,
):
    vision_model = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    if Path(backbone_weights).exists() and not validate_checkpoint_file(
        backbone_weights, expected_sha256=None
    ):
        print(f"Warning: checkpoint at {backbone_weights} looks corrupt, re-downloading")
        Path(backbone_weights).unlink(missing_ok=True)

    backbone_weights = ensure_backbone_checkpoint(backbone_weights)
    load_checkpoint(vision_model, backbone_weights)
    vision_model.to(device)
    vision_model.eval()
    for param in vision_model.parameters():
        param.requires_grad = False

    llm = load_minicpm_llm(
        device,
        model_name=llm_model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        adapter_path=adapter_path,
    )

    model = DINOv3MiniCPMHybrid(
        vision_model=vision_model,
        llm=llm,
        num_frames=num_frames,
    ).to(device)

    if hasattr(model.llm, "enable_input_require_grads"):
        model.llm.enable_input_require_grads()

    tokenizer = load_minicpm_tokenizer(llm_model_name)
    return model, tokenizer


def adapter_trainable_params(model: DINOv3MiniCPMHybrid):
    return [p for p in model.llm.parameters() if p.requires_grad]


def vision_adapter_params(model: DINOv3MiniCPMHybrid):
    return list(model.vision_projection.parameters()) + list(
        model.projection_norm.parameters()
    )


def save_hybrid_checkpoint(model: DINOv3MiniCPMHybrid, checkpoint_dir: str) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(path / "adapter")
    torch.save(
        {
            "vision_projection": model.vision_projection.state_dict(),
            "projection_norm": model.projection_norm.state_dict(),
            "num_frames": model.num_frames,
            "num_visual_tokens": model.num_visual_tokens,
            "architecture": "video_cls",
        },
        path / "vision_adapter.pt",
    )


def load_hybrid_checkpoint(
    model: DINOv3MiniCPMHybrid,
    checkpoint_dir: str,
    device: torch.device,
    trainable_adapter: bool = False,
) -> None:
    from peft import PeftModel

    path = Path(checkpoint_dir)
    adapter_dir = path / "adapter"
    vision_path = path / "vision_adapter.pt"

    if adapter_dir.exists():
        if isinstance(model.llm, PeftModel):
            base_model = model.llm.get_base_model()
        else:
            base_model = model.llm
        model.llm = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=trainable_adapter,
        ).to(device)

    if vision_path.exists():
        state = torch.load(vision_path, map_location=device, weights_only=True)
        arch = state.get("architecture")
        if arch is not None and arch != "video_cls":
            raise RuntimeError(
                f"Checkpoint at {checkpoint_dir} has architecture={arch!r}, "
                "expected 'video_cls'. Image-VQA adapters are not compatible."
            )
        ckpt_frames = state.get("num_frames", state.get("num_visual_tokens"))
        if ckpt_frames is not None and int(ckpt_frames) != model.num_frames:
            print(
                f"Warning: checkpoint num_frames={ckpt_frames} != model "
                f"num_frames={model.num_frames}; using model setting."
            )
        model.vision_projection.load_state_dict(state["vision_projection"])
        model.projection_norm.load_state_dict(state["projection_norm"])


if __name__ == "__main__":
    from dinov3.utils.device import get_device
    from heads.vqa.minicpm_loader import tokenize_chat_pair

    device = get_device()
    num_frames = DEFAULT_NUM_FRAMES
    model, tokenizer = build_hybrid_model(device, num_frames=num_frames)

    videos = torch.randn(2, num_frames, 3, IMG_SIZE, IMG_SIZE, device=device)
    questions = [
        "Does this video contain any potentially violent or criminal activities?",
        "What type of abnormal event is present in the video?",
    ]
    answers = [
        "Yes, a fight is taking place near the entrance.",
        "Fighting between two people in a hallway.",
    ]

    input_ids = []
    labels = []
    for question, answer in zip(questions, answers):
        ids, lbls = tokenize_chat_pair(tokenizer, question, answer, max_length=128)
        input_ids.append(ids)
        labels.append(lbls)

    max_len = max(len(row) for row in input_ids)
    pad_id = tokenizer.pad_token_id
    input_ids_t = torch.full((2, max_len), pad_id, dtype=torch.long, device=device)
    labels_t = torch.full((2, max_len), -100, dtype=torch.long, device=device)
    attn = torch.zeros((2, max_len), dtype=torch.long, device=device)
    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        input_ids_t[i, : len(ids)] = torch.tensor(ids, device=device)
        labels_t[i, : len(lbls)] = torch.tensor(lbls, device=device)
        attn[i, : len(ids)] = 1

    visual = model.encode_visual(videos)
    print("visual_embeds:", tuple(visual.shape))
    out = model(videos, input_ids_t, attn, labels=labels_t)
    print("loss:", out.loss.item())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.1f}M")
