"""Runs the three layers and combines them into one verdict.

When the LLM answers, it decides - it is the only layer that reads intent and context. The toxicity model can
still overrule it on a strong abuse score. Without the LLM, simple rules over the BERT scores decide.
Swearing alone never makes a comment abusive: it adds a "profanity" flag and the word is censored.
"""

import time

from .bert_models import models
from .config import settings
from .llm import judge
from .profanity_filter import check_profanity
from .schemas import AnalyzeResponse, LlmResult

LLM_VERDICTS = {"clean": "clean", "negative_feedback": "negative", "abusive": "abusive"}


def analyze(text: str, use_llm: bool = True) -> AnalyzeResponse:
    started = time.perf_counter()

    profanity = check_profanity(text)
    toxicity = models.toxicity(text)
    sentiment = models.sentiment(text)
    llm = judge.judge(text) if use_llm else LlmResult(status="skipped", message="LLM check turned off for this request.")

    reasons = []
    abuse = f"{toxicity.abuse_label.replace('_', ' ')} {toxicity.abuse_score:.0%}"

    if llm.judgement:
        verdict, decided_by = LLM_VERDICTS[llm.judgement.verdict], "llm"
        reasons.append(f"LLM: {llm.judgement.reason}")
        if verdict != "abusive" and toxicity.abuse_score >= settings.abuse_override_threshold:
            verdict, decided_by = "abusive", "llm+toxicity-override"
            reasons.append(f"Overruled by the toxicity model: {abuse}")
    else:
        decided_by = "rules"
        if toxicity.is_abusive:
            verdict = "abusive"
            reasons.append(f"Toxicity model: {abuse}")
        elif toxicity.toxic_score >= settings.toxicity_threshold and not profanity.found:
            # Hostile wording with no swear word to explain the score - typical of hate and harassment.
            verdict = "abusive"
            reasons.append(f"Toxicity model: hostile wording (toxic {toxicity.toxic_score:.0%}) without swearing")
        elif sentiment.is_negative:
            verdict = "negative"
            reasons.append(f"Sentiment model: negative ({sentiment.scores['negative']:.0%})")
        else:
            verdict = "clean"
            reasons.append("No negativity or abuse found.")

    flags = ["profanity"] if profanity.found else []
    if profanity.found:
        reasons.append(f"Profanity censored: {', '.join(profanity.words) or 'yes'}")

    return AnalyzeResponse(
        text=text,
        verdict=verdict,
        flags=flags,
        decided_by=decided_by,
        reasons=reasons,
        profanity=profanity,
        toxicity=toxicity,
        sentiment=sentiment,
        llm=llm,
        device=models.device,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
