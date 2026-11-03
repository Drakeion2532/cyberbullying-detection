# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/cyberbullying-detection · MIT licence · sp2532-f30cd46f33af
"""Classify text from the command line.

    python -m cbd.predict "go back to your country"
"""
from __future__ import annotations

import sys

import joblib

from . import MODELS

LABELS = {"age": "age-based bullying", "ethnicity": "ethnicity / race-based bullying",
          "gender": "gender / sexuality-based bullying", "religion": "religion-based bullying",
          "other_cyberbullying": "other bullying", "not_cyberbullying": "not cyberbullying"}


def load(task="five_class"):
    return joblib.load(MODELS / f"ensemble_{task}.joblib")


def classify(model, text: str) -> dict:
    p = model.predict_proba([text])[0]
    order = p.argsort()[::-1]
    return {"label": model.classes[order[0]], "confidence": float(p[order[0]]),
            "probabilities": {model.classes[i]: round(float(p[i]), 4) for i in order},
            "votes": model.explain(text)}


def main():
    model = load()
    for text in sys.argv[1:] or ["have a great day everyone"]:
        r = classify(model, text)
        print(f"{LABELS[r['label']]:35s} {r['confidence']:.0%}  <- {text}")


if __name__ == "__main__":
    main()
