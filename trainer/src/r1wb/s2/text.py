"""Sentence embeddings of instructions for the small System 2 (a frozen multilingual sentence encoder).

The encoder is pretrained on many languages, so wordings that never occur in the training phrases ("좌측으로" for
"왼쪽으로", "halt" for "stop") land near the ones that do: that is where the small model's understanding of held-out
phrasings comes from. Embeddings are cached in a file next to the data (phrase -> vector).
"""
from __future__ import annotations

from pathlib import Path

import torch

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WIDTH = 384


class TextEncoder:
  def __init__(self, cache: Path | None = None, device: str = "cpu"):
    self.cache_path, self.device = cache, device
    self.cache: dict[str, torch.Tensor] = torch.load(cache, weights_only=False) if cache and cache.exists() else {}
    self._model = None

  def _load(self):
    from transformers import AutoModel, AutoTokenizer
    self._tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    self._model = AutoModel.from_pretrained(ENCODER).to(self.device).eval()

  def __call__(self, phrases: list[str]) -> torch.Tensor:
    missing = [p for p in dict.fromkeys(phrases) if p not in self.cache]
    if missing:
      if self._model is None:
        self._load()
      with torch.no_grad():
        for begin in range(0, len(missing), 64):
          part = missing[begin:begin + 64]
          batch = self._tokenizer(part, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
          hidden = self._model(**batch).last_hidden_state
          mask = batch["attention_mask"].unsqueeze(-1).float()
          pooled = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
          for phrase, vector in zip(part, pooled.cpu()):
            self.cache[phrase] = vector
      if self.cache_path:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cache, self.cache_path)
    return torch.stack([self.cache[p] for p in phrases])
