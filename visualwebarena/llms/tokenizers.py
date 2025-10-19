from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore


TIKTOKEN_MODEL_ALIASES: dict[str, str] = {
    # deepseek 系列不在 tiktoken 的默认映射表，回退到与 GPT-4 系列兼容的编码器。
    "deepseek-v3-250324": "cl100k_base",
}


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            encoding_name = TIKTOKEN_MODEL_ALIASES.get(model_name, model_name)
            if encoding_name == model_name:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            else:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
