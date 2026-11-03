"""Layer 1: a fast bad-word list check (also catches simple l33t spellings like 'sh1t')."""

import re

from better_profanity import profanity

from .schemas import ProfanityResult

profanity.load_censor_words()

_WORD = re.compile(r"[\w@$!*]+")
# Self-censored swearing the word list can't match, like "f***" or "sh*t".
_MASKED = re.compile(r"\b[a-z]{1,4}\*{1,4}[a-z]{0,4}(?![\w*])", re.IGNORECASE)


def check_profanity(text: str) -> ProfanityResult:
    words = {w for w in _WORD.findall(text) if profanity.contains_profanity(w)}
    words |= {m for m in _MASKED.findall(text) if len(m) >= 4}
    return ProfanityResult(
        found=bool(words) or profanity.contains_profanity(text),
        words=sorted(words, key=str.lower),
        censored_text=profanity.censor(text),
    )
