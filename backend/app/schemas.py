from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["clean", "negative", "abusive"]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    use_llm: bool = True


class ProfanityResult(BaseModel):
    found: bool
    words: list[str]
    censored_text: str


class ToxicityResult(BaseModel):
    is_abusive: bool
    abuse_label: str
    abuse_score: float
    toxic_score: float
    scores: dict[str, float]


class SentimentResult(BaseModel):
    label: Literal["negative", "neutral", "positive"]
    is_negative: bool
    scores: dict[str, float]


class LlmJudgement(BaseModel):
    """The structured answer the LLM must return."""

    verdict: Literal["clean", "negative_feedback", "abusive"]
    categories: list[
        Literal["insult", "profanity", "hate", "threat", "harassment", "sexual", "complaint", "sarcasm", "other"]
    ]
    reason: str


class LlmResult(BaseModel):
    status: Literal["ok", "disabled", "skipped", "refused", "error"]
    model: str | None = None
    judgement: LlmJudgement | None = None
    message: str | None = None


class AnalyzeResponse(BaseModel):
    text: str
    verdict: Verdict
    flags: list[Literal["profanity"]]
    decided_by: Literal["llm", "llm+toxicity-override", "rules"]
    reasons: list[str]
    profanity: ProfanityResult
    toxicity: ToxicityResult
    sentiment: SentimentResult
    llm: LlmResult
    device: str
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    device: str
    gpu_name: str | None
    llm_configured: bool
    llm_model: str
