"""Download COCO 2017 detection data for DETR training."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
IMAGE_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}
IMAGE_SIZE_HINT = {
    "train2017": "~18GB",
    "val2017": "~1GB",
}


def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"Using cached download: {dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    last_pct = -1

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        nonlocal last_pct
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(100 * downloaded / total_size))
        if pct != last_pct and pct % 5 == 0:
            last_pct = pct
            print(
                f"  {pct:3d}% "
                f"({min(downloaded, total_size) / 1e9:.2f}/"
                f"{total_size / 1e9:.2f} GB)",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, filename=tmp, reporthook=_reporthook)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _safe_extract(zip_path: Path, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        for member in archive.infolist():
            target = (dest_dir / member.filename).resolve()
            if not str(target).startswith(str(dest_dir)):
                raise RuntimeError(f"Zip slip detected for member: {member.filename}")
            members.append(member)
        archive.extractall(dest_dir, members=members)


def ensure_coco_annotations(coco_root: str | Path) -> Path:
    """Download and extract COCO train/val annotations under ``coco_root``."""
    root = Path(coco_root)
    ann_dir = root / "annotations"
    needed = [
        ann_dir / "instances_train2017.json",
        ann_dir / "instances_val2017.json",
    ]
    if all(path.is_file() for path in needed):
        return ann_dir

    zip_path = _download_file(
        ANNOTATIONS_URL, root / "annotations_trainval2017.zip"
    )
    print(f"Extracting {zip_path.name} -> {root}")
    _safe_extract(zip_path, root)
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "COCO annotations zip extracted but expected files are missing: "
            + ", ".join(missing)
        )
    return ann_dir


def ensure_coco_images(coco_root: str | Path, split: str) -> Path:
    """Download and extract a COCO image split under ``coco_root/images/<split>``."""
    if split not in IMAGE_URLS:
        raise ValueError(
            f"Unsupported COCO split {split!r}; expected one of {sorted(IMAGE_URLS)}"
        )

    root = Path(coco_root)
    img_dir = root / "images" / split
    if img_dir.is_dir() and any(img_dir.iterdir()):
        return img_dir

    hint = IMAGE_SIZE_HINT.get(split, "")
    print(f"COCO {split} images missing under {img_dir} ({hint})")
    zip_path = _download_file(IMAGE_URLS[split], root / f"{split}.zip")
    images_root = root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {images_root}")
    _safe_extract(zip_path, images_root)
    if not img_dir.is_dir() or not any(img_dir.iterdir()):
        raise FileNotFoundError(
            f"COCO {split} zip extracted but image directory is empty: {img_dir}"
        )
    return img_dir


def ensure_coco_split(img_dir: str | Path, ann_file: str | Path) -> None:
    """Ensure one COCO split exists for the given image/annotation paths.

    Expects the usual layout::

        <coco_root>/images/<split>/
        <coco_root>/annotations/instances_<split>.json
    """
    img_path = Path(img_dir)
    ann_path = Path(ann_file)
    split = img_path.name
    coco_root = img_path.parent.parent

    ann_ready = ann_path.is_file()
    img_ready = img_path.is_dir() and any(img_path.iterdir())
    if ann_ready and img_ready:
        return

    print(f"Preparing COCO {split} under {coco_root}")
    ensure_coco_annotations(coco_root)
    ensure_coco_images(coco_root, split)

    if not Path(ann_file).is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
    if not Path(img_dir).is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {img_dir}")
