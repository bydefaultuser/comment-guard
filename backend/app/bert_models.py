"""Layer 2: BERT-family models, run on the GPU when one is available."""

import threading

import torch
from transformers import pipeline

from .config import settings
from .schemas import SentimentResult, ToxicityResult

TOXICITY_MODEL = "unitary/toxic-bert"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# toxic-bert's "toxic" and "obscene" labels fire on any swearing, even in praise ("holy shit, this is amazing").
# These labels are the ones that mean someone is being attacked.
ABUSE_LABELS = ("insult", "threat", "identity_hate", "severe_toxic")


class BertModels:
    def __init__(self) -> None:
        self.use_gpu = torch.cuda.is_available() and not settings.force_cpu
        self.device = "cuda" if self.use_gpu else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.use_gpu else None
        self._lock = threading.Lock()
        self._toxicity = None
        self._sentiment = None

    @property
    def loaded(self) -> bool:
        return self._toxicity is not None and self._sentiment is not None

    def load(self) -> None:
        """Downloads the models on first run (~1 GB), then loads them from the local cache."""
        device = 0 if self.use_gpu else -1
        self._toxicity = pipeline(
            "text-classification", model=TOXICITY_MODEL, device=device, top_k=None, function_to_apply="sigmoid"
        )
        self._sentiment = pipeline("text-classification", model=SENTIMENT_MODEL, device=device, top_k=None)

    def toxicity(self, text: str) -> ToxicityResult:
        with self._lock:
            raw = self._toxicity(text, truncation=True, max_length=512)
        scores = {item["label"]: round(float(item["score"]), 4) for item in _first(raw)}
        abuse_label = max(ABUSE_LABELS, key=lambda label: scores.get(label, 0.0))
        return ToxicityResult(
            is_abusive=scores.get(abuse_label, 0.0) >= settings.toxicity_threshold,
            abuse_label=abuse_label,
            abuse_score=scores.get(abuse_label, 0.0),
            toxic_score=scores.get("toxic", 0.0),
            scores=scores,
        )

    def sentiment(self, text: str) -> SentimentResult:
        with self._lock:
            raw = self._sentiment(text, truncation=True, max_length=512)
        scores = {item["label"].lower(): round(float(item["score"]), 4) for item in _first(raw)}
        return SentimentResult(
            label=max(scores, key=scores.get),
            is_negative=scores.get("negative", 0.0) >= settings.negative_threshold,
            scores=scores,
        )


def _first(raw):
    # Pipelines return [[{label, score}, ...]] for a single string with top_k=None on some versions.
    return raw[0] if raw and isinstance(raw[0], list) else raw


models = BertModels()
