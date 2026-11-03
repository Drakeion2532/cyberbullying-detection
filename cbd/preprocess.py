# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/cyberbullying-detection · MIT licence · sp2532-f30cd46f33af
"""Tweet cleaning: the same steps for training and for live predictions.

1. lower-case; strip URLs, @mentions, the 'RT' prefix and HTML entities
2. keep the word inside hashtags (#muslim -> muslim), drop emojis and digits
3. expand common contractions and keep negations ("not", "no", "never"),
   because "you are not stupid" and "you are stupid" must stay different
4. remove other stopwords and Porter-stem what is left
"""
from __future__ import annotations

import html
import re

from nltk.stem import PorterStemmer          # rule-based, needs no downloaded data
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stem = PorterStemmer().stem
NEGATIONS = {"no", "not", "nor", "never", "cannot"}
STOP = (set(ENGLISH_STOP_WORDS) | {"rt", "amp", "im", "u", "ur"}) - NEGATIONS
CONTRACTIONS = {"n't": " not", "'re": " are", "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am"}

_url = re.compile(r"https?://\S+|www\.\S+")
_mention = re.compile(r"@\w+")
_nonalpha = re.compile(r"[^a-z\s]")
_space = re.compile(r"\s+")


def clean(text: str) -> str:
    t = html.unescape(str(text)).lower()
    t = _url.sub(" ", t)
    t = _mention.sub(" ", t)
    t = t.replace("#", " ")
    for k, v in CONTRACTIONS.items():
        t = t.replace(k, v)
    t = _nonalpha.sub(" ", t)                  # removes emojis, digits and punctuation
    return _space.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in clean(text).split() if w not in STOP and len(w) > 1]


def normalise(text: str) -> str:
    """Cleaned, stop-word-free, stemmed text as one string (input to TF-IDF)."""
    return " ".join(tokens(text))
