import argparse
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from dinov3.checkpoints.load import load_checkpoint
from dinov3.models import vit_small
from dinov3.utils.device import get_device
from heads.detr.dataset import make_dataloader
from heads.detr.matcher import HungarianLoss
from heads.detr.transformer import DETR, build_detr
from logger import SQLiteLogger

BACKBONE_WEIGHTS = (
    "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
DEFAULT_OUT_PATH = "dinov3/checkpoints/model/detr_decoder.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 + DETR decoder")
    parser.add_argument(
        "--img-dir", type=str, default="coco/images/val2017"
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default="coco/annotations/instances_val2017.json",
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


def train_step(model: DETR, img_embed, target, lf: HungarianLoss):
    out = model(img_embed)
    loss, stats = lf(out, target)
    return loss, stats


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    loader, num_batches = make_dataloader(
        args.img_dir,
        args.ann_file,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
    )
    print("Total batches per epoch:", num_batches)

    dinov3_small = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    load_checkpoint(dinov3_small, BACKBONE_WEIGHTS)
    dinov3_small.to(device)
    dinov3_small.eval()
    for p in dinov3_small.parameters():
        p.requires_grad = False

    total = sum(p.numel() for p in dinov3_small.parameters())
    print(f"Total backbone parameters: {total / 1e6:.1f}M")

    detr_decoder = build_detr(
        d_model=384,
        num_layers=5,
        n_classes=92,
        n_points=3,
    ).to(device)

    out_path = args.output
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

    try:
        for epoch in range(args.epochs):
            total_loss = 0.0
            prog_bar = tqdm(
                loader, desc="Training", unit="batch", total=num_batches
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

            epoch_loss = total_loss / num_batches
            print(f"Epoch {epoch}, Loss: {epoch_loss}")
            if logger is not None:
                logger.log_metric(
                    "train/loss",
                    epoch_loss,
                    epoch=epoch + 1,
                    split="train",
                )
            torch.save(detr_decoder.state_dict(), out_path)

        if logger is not None:
            logger.finish(status="completed")
    except Exception:
        if logger is not None:
            logger.finish(status="failed")
        raise
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
