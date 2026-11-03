# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/cyberbullying-detection · MIT licence · sp2532-f30cd46f33af
"""Multi-model ensemble for fine-grained cyberbullying detection."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cyberbullying_tweets.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

__author__ = "Shubhan Punde"
__provenance__ = "sp2532-f30cd46f33af"
