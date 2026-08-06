import argparse
import os
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from dinov3.checkpoints.load import (
    ensure_backbone_checkpoint,
    load_checkpoint,
    validate_checkpoint_file,
)
from dinov3.models import vit_small
from dinov3.utils.device import get_device
from heads.detr.dataset import make_dataloader
from heads.detr.download import ensure_coco_split
from heads.detr.matcher import HungarianLoss
from heads.detr.transformer import DETR, build_detr
from logger import SQLiteLogger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COCO_ROOT = _REPO_ROOT / "coco"
BACKBONE_WEIGHTS = str(
    _REPO_ROOT
    / "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
DEFAULT_OUT_PATH = str(_REPO_ROOT / "dinov3/checkpoints/model/detr_decoder.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 + DETR decoder")
    parser.add_argument(
        "--img-dir",
        type=str,
        default=str(_COCO_ROOT / "images" / "train2017"),
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default=str(_COCO_ROOT / "annotations" / "instances_train2017.json"),
    )
    parser.add_argument(
        "--val-img-dir",
        type=str,
        default=str(_COCO_ROOT / "images" / "val2017"),
    )
    parser.add_argument(
        "--val-ann-file",
        type=str,
        default=str(_COCO_ROOT / "annotations" / "instances_val2017.json"),
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=BACKBONE_WEIGHTS,
        help="Path to DINOv3 ViT-S/16 weights (auto-downloads if missing)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-download missing backbone weights or COCO data",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--log-db",
        type=str,
        default=None,
        help="SQLite database path for training metrics (optional)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for the logged training run",
    )
    return parser.parse_args()


def _load_backbone(weights: str, device, auto_download: bool = True):
    model = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    if Path(weights).exists() and not validate_checkpoint_file(
        weights, expected_sha256=None
    ):
        print(
            f"Warning: checkpoint at {weights} looks corrupt, re-downloading"
        )
        Path(weights).unlink(missing_ok=True)
    weights = ensure_backbone_checkpoint(weights, auto_download=auto_download)
    load_checkpoint(model, weights)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def train_step(model: DETR, img_embed, target, lf: HungarianLoss):
    out = model(img_embed)
    loss, stats = lf(out, target)
    return loss, stats


@torch.no_grad()
def evaluate(model: DETR, backbone, loader, lf: HungarianLoss, device, num_batches: int):
    model.eval()
    total_loss = 0.0
    prog_bar = tqdm(
        loader, desc="Validating", unit="batch", total=num_batches, leave=False
    )
    for batch in prog_bar:
        image = batch["image"].to(device)
        target = {
            "boxes": batch["boxes"].to(device),
            "labels": batch["labels"].to(device),
        }

        features = backbone(image, masks=None, is_training=True)
        patches = features["x_norm_patchtokens"]
        loss, _ = train_step(model, patches, target, lf)

        prog_bar.set_postfix(loss=f"{loss.item():.4f}")
        total_loss += loss.item()
    prog_bar.close()
    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    auto_download = not args.no_download
    if auto_download:
        ensure_coco_split(args.img_dir, args.ann_file)
        ensure_coco_split(args.val_img_dir, args.val_ann_file)

    dinov3_small = _load_backbone(
        args.backbone, device, auto_download=auto_download
    )
    total = sum(p.numel() for p in dinov3_small.parameters())
    print(f"Total backbone parameters: {total / 1e6:.1f}M")

    train_loader, num_train_batches = make_dataloader(
        args.img_dir,
        args.ann_file,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader, num_val_batches = make_dataloader(
        args.val_img_dir,
        args.val_ann_file,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print("Train batches per epoch:", num_train_batches)
    print("Val batches:", num_val_batches)

    detr_decoder = build_detr(
        d_model=384,
        num_layers=6,
        n_classes=92,
        n_heads=8,
        n_queries=50,
        n_points=4,
    ).to(device)

    out_path = args.output
    best_path = str(Path(out_path).with_name(f"{Path(out_path).stem}_best.pt"))
    if os.path.exists(out_path):
        state_dict = torch.load(out_path)
        detr_decoder.load_state_dict(state_dict)

    total = sum(p.numel() for p in detr_decoder.parameters())
    print(f"Total Decoder parameters: {total / 1e6:.1f}M")

    optimizer = optim.AdamW(
        detr_decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lf = HungarianLoss(num_classes=91)

    logger = None
    if args.log_db:
        logger = SQLiteLogger(
            args.log_db,
            head="detr",
            name=args.run_name,
            config=vars(args),
        )
        print(f"Logging run {logger.run_id} to {args.log_db}")

    best_val_loss = float("inf")

    try:
        for epoch in range(args.epochs):
            detr_decoder.train()
            total_loss = 0.0
            prog_bar = tqdm(
                train_loader, desc="Training", unit="batch", total=num_train_batches
            )
            for batch in prog_bar:
                image = batch["image"].to(device)
                target = {
                    "boxes": batch["boxes"].to(device),
                    "labels": batch["labels"].to(device),
                }

                with torch.no_grad():
                    features = dinov3_small(image, masks=None, is_training=True)
                    patches = features["x_norm_patchtokens"]

                optimizer.zero_grad()
                loss, _ = train_step(detr_decoder, patches, target, lf)
                loss.backward()
                optimizer.step()

                prog_bar.set_postfix(loss=f"{loss.item():.4f}")
                total_loss += loss.item()
            prog_bar.close()

            train_loss = total_loss / max(num_train_batches, 1)
            val_loss = evaluate(
                detr_decoder,
                dinov3_small,
                val_loader,
                lf,
                device,
                num_val_batches,
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
            if logger is not None:
                logger.log_metrics(
                    {
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                    },
                    epoch=epoch + 1,
                )

            torch.save(detr_decoder.state_dict(), out_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(detr_decoder.state_dict(), best_path)
                print(f"  saved best checkpoint (val_loss={best_val_loss:.4f})")

        if logger is not None:
            logger.finish(status="completed")
        print(
            f"Training complete. Best val_loss={best_val_loss:.4f}. "
            f"Latest: {out_path}, best: {best_path}"
        )
    except Exception:
        if logger is not None:
            logger.finish(status="failed")
        raise
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
