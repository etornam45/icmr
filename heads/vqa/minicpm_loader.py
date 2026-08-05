"""Backward-compatible re-exports; prefer heads.vqa.llm_loader."""

from heads.vqa.llm_loader import (  # noqa: F401
    LLM_MODEL_NAME as MINICPM_MODEL_NAME,
    apply_chat,
    load_llm as load_minicpm_llm,
    load_llm_tokenizer as load_minicpm_tokenizer,
    tokenize_chat_pair,
)
