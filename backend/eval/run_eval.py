"""Scores every layer against labelled comments.

Run from the backend folder:  .\\.venv\\Scripts\\python.exe -m eval.run_eval [comments.json|holdout.json]
Uses the Bedrock LLM when it is configured in .env (one short call per comment).
holdout.json was written before the rules were tuned on comments.json - never tune on it.
"""

import json
import sys
from pathlib import Path

from app.analyzer import analyze
from app.bert_models import models


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "comments.json"
    cases = json.loads((Path(__file__).parent / name).read_text(encoding="utf-8"))
    models.load()

    hits = {"final": 0, "llm": 0, "rules": 0, "profanity": 0}
    llm_runs = 0
    for case in cases:
        result = analyze(case["text"])
        rules = analyze(case["text"], use_llm=False).verdict
        llm_verdict = result.llm.judgement.verdict.replace("_feedback", "") if result.llm.judgement else None

        hits["final"] += result.verdict == case["verdict"]
        hits["rules"] += rules == case["verdict"]
        hits["profanity"] += result.profanity.found == case["profanity"]
        if llm_verdict:
            llm_runs += 1
            hits["llm"] += llm_verdict == case["verdict"]

        mark = "OK " if result.verdict == case["verdict"] else "XX "
        swear = "" if result.profanity.found == case["profanity"] else f"  (profanity flag {result.profanity.found})"
        print(
            f"{mark} want {case['verdict']:<8} got {result.verdict:<8} llm {llm_verdict or result.llm.status:<8} "
            f"rules {rules:<8} {case['text'][:55]}{swear}"
        )

    n = len(cases)
    print(f"\nfinal verdict   {hits['final']}/{n}")
    print(f"llm alone       {hits['llm']}/{llm_runs}" if llm_runs else "llm alone       not configured")
    print(f"rules (no llm)  {hits['rules']}/{n}")
    print(f"profanity flag  {hits['profanity']}/{n}")


if __name__ == "__main__":
    main()
