"""BPE tokenizer for VAU-Bench video descriptions.

Trains once from train-split captions (if no checkpoint exists), then encodes /
decodes text for ``CaptionHead``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

DEFAULT_CACHE_DIR = "data/VAU-Bench"
DEFAULT_TOKENIZER_DIR = "dinov3/checkpoints/model/caption/tokenizer"
DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MAX_LENGTH = 128
HF_REPO_ID = "7xiang/VAU-Bench"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

COL_DESCRIPTION = "Description"
COL_VIDEO = "Video Name"
COL_ANOMALY_CLASS = "Anomaly Class"

_UCF_CLASS_RE = re.compile(
    r"^(?:ucf_)?(?:"
    r"Abuse|Arrest|Arson|Assault|Burglary|Explosion|Fighting|"
    r"RoadAccidents|Robbery|Shooting|Shoplifting|Stealing|Vandalism"
    r")\d+_x264\.mp4$",
    re.IGNORECASE,
)
_UCF_NORMAL_RE = re.compile(
    r"^(?:ucf_)?Normal_Videos_\d+_x264\.mp4$",
    re.IGNORECASE,
)


class CaptionTokenizer:
    """Thin wrapper around a HuggingFace ``tokenizers`` BPE model."""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
        self.unk_token_id = tokenizer.token_to_id(UNK_TOKEN)
        self.bos_token_id = tokenizer.token_to_id(BOS_TOKEN)
        self.eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
        if None in (
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        ):
            raise RuntimeError("Tokenizer is missing required special tokens")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = DEFAULT_MAX_LENGTH,
        truncation: bool = True,
    ) -> list[int]:
        """Encode a caption to token ids (optionally with BOS/EOS)."""
        if add_special_tokens:
            ids = self._tok.encode(text).ids
        else:
            post = self._tok.post_processor
            self._tok.post_processor = None
            try:
                ids = self._tok.encode(text).ids
            finally:
                self._tok.post_processor = post

        if max_length is not None and truncation and len(ids) > max_length:
            if add_special_tokens and len(ids) >= 2:
                ids = ids[: max_length - 1] + [self.eos_token_id]
            else:
                ids = ids[:max_length]
        return ids

    def id_to_token(self, token_id: int) -> str:
        token = self._tok.id_to_token(token_id)
        return token if token is not None else self.unk_token

    def tokenize(self, text: str, *, add_special_tokens: bool = True) -> list[str]:
        """Return BPE pieces for ``text`` (Ġ = leading space in byte-level BPE)."""
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        text = self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
        return text.strip()

    def batch_encode(
        self,
        texts: Sequence[str],
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        add_special_tokens: bool = True,
        return_tensors: bool = True,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        """Pad a batch of captions (right-padded)."""
        encoded = [
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=True,
            )
            for text in texts
        ]
        batch_max = min(max_length, max((len(row) for row in encoded), default=1))
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for row in encoded:
            pad_len = batch_max - len(row)
            input_ids.append(row + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)

        if not return_tensors:
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path / "tokenizer.json"))
        meta = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "max_length": DEFAULT_MAX_LENGTH,
        }
        (path / "tokenizer_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> CaptionTokenizer:
        path = Path(directory)
        tok_path = path / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"No tokenizer.json under {path}")
        return cls(Tokenizer.from_file(str(tok_path)))


def _video_source(video_name: str) -> str | None:
    name = Path(video_name).name
    if name.startswith("ucf_") or _UCF_CLASS_RE.match(name) or _UCF_NORMAL_RE.match(name):
        return "ucf"
    if name.startswith("msad_"):
        return "msad"
    if name.startswith("ecva_"):
        return "ecva"
    return None


def _collect_descriptions(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    sources: Sequence[str] | str | None = "ucf",
    splits: Sequence[str] = ("train",),
) -> list[str]:
    if isinstance(sources, str):
        source_set = {
            part.strip().casefold()
            for part in sources.split(",")
            if part.strip()
        } or None
    elif sources is None:
        source_set = None
    else:
        source_set = {str(s).strip().casefold() for s in sources if str(s).strip()}

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    seen: set[str] = set()

    for split in splits:
        split_name = "validation" if split == "val" else split
        dataset = load_dataset(
            HF_REPO_ID,
            split=split_name,
            cache_dir=str(cache_root / "hf_cache"),
        )
        for row in dataset:
            video_name = str(row.get(COL_VIDEO, "") or "")
            description = str(row.get(COL_DESCRIPTION, "") or "").strip()
            anomaly_class = str(row.get(COL_ANOMALY_CLASS, "") or "").strip()
            if not description or not anomaly_class:
                continue
            if anomaly_class in {"-1", "None", "N/A", "n/a"}:
                continue
            source = _video_source(video_name)
            if source_set is not None and source not in source_set:
                continue
            if description in seen:
                continue
            seen.add(description)
            texts.append(description)

    if not texts:
        raise RuntimeError(
            f"No VAU-Bench descriptions found under {cache_dir} "
            f"(splits={list(splits)}, sources={sources}). "
            "Run: python -m heads.caption.vau_dataset --annotations-only --split train"
        )
    return texts


def train_tokenizer(
    texts: Sequence[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> CaptionTokenizer:
    """Train a byte-level BPE tokenizer on caption strings."""
    if vocab_size < len(SPECIAL_TOKENS) + 256:
        raise ValueError(
            f"vocab_size={vocab_size} is too small; need at least "
            f"{len(SPECIAL_TOKENS) + 256}"
        )

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )
    return CaptionTokenizer(tokenizer)


def ensure_tokenizer(
    tokenizer_dir: str | Path = DEFAULT_TOKENIZER_DIR,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    sources: Sequence[str] | str | None = "ucf",
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    force_retrain: bool = False,
) -> CaptionTokenizer:
    """Load an existing caption tokenizer, or train + save if missing.

    Call this before caption-model training so ``CaptionHead`` gets a fixed vocab.
    """
    path = Path(tokenizer_dir)
    tok_file = path / "tokenizer.json"
    if tok_file.exists() and not force_retrain:
        print(f"Loading caption tokenizer from {path}")
        return CaptionTokenizer.load(path)

    print(
        f"Training caption tokenizer (vocab_size={vocab_size}) "
        f"from VAU-Bench descriptions → {path}"
    )
    texts = _collect_descriptions(cache_dir=cache_dir, sources=sources)
    print(f"  corpus: {len(texts)} unique descriptions")
    tokenizer = train_tokenizer(texts, vocab_size=vocab_size)
    tokenizer.save(path)
    print(
        f"  saved vocab_size={tokenizer.vocab_size} "
        f"(pad={tokenizer.pad_token_id}, bos={tokenizer.bos_token_id}, "
        f"eos={tokenizer.eos_token_id})"
    )
    return tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or load a BPE tokenizer on VAU-Bench descriptions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TOKENIZER_DIR,
        help="Directory for tokenizer.json",
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--sources",
        type=str,
        default="ucf",
        help="Comma-separated sources (default: ucf). Empty = all.",
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if tokenizer.json already exists",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="A person enters the store and starts fighting near the counter.",
        help="Optional string to encode/decode as a sanity check",
    )
    args = parser.parse_args()

    sources = args.sources if args.sources else None
    tok = ensure_tokenizer(
        args.output,
        cache_dir=args.cache_dir,
        sources=sources,
        vocab_size=args.vocab_size,
        force_retrain=args.force,
    )
    ids = tok.encode(args.demo)
    pieces = tok.tokenize(args.demo)
    print(f"demo text: {args.demo!r}")
    print(f"demo split ({len(pieces)} tokens):")
    for index, (piece, token_id) in enumerate(zip(pieces, ids)):
        print(f"  [{index:02d}] {token_id:>5}  {piece!r}")
    print(f"demo decode: {tok.decode(ids)!r}")
