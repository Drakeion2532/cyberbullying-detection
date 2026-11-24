# Copyright (c) 2026 Shubhan Punde · MIT licence
"""Load and split the public cyberbullying tweet dataset.

Dataset: 47,692 tweets labelled religion, age, gender, ethnicity, other_cyberbullying
and not_cyberbullying (Wang, Fu & Lu, "SOSNet", IEEE BigData 2020; published on Kaggle as
"Cyberbullying Classification"). 1,675 tweets appear more than once, sometimes with
different labels, so duplicates are removed *before* splitting; otherwise the same tweet
can sit in train and test and inflate the score.

Two task set-ups are supported:
  five_class (default): the four targeted categories plus not_cyberbullying. The paper
      reports results per targeted category (gender, religion, age, ethnicity).
  six_class: also keeps other_cyberbullying, a vague catch-all that is much harder.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import DATA

TARGETED = ["age", "ethnicity", "gender", "religion"]


def load(task: str = "five_class", path=DATA) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"tweet_text": "text", "cyberbullying_type": "label"})
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    conflicting = df.groupby("text")["label"].nunique()
    df = df[~df["text"].isin(conflicting[conflicting > 1].index)]     # same tweet, different labels
    df = df.drop_duplicates("text")
    if task == "five_class":
        df = df[df["label"] != "other_cyberbullying"]
    elif task != "six_class":
        raise ValueError(task)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size=0.2, seed=42):
    return train_test_split(df["text"], df["label"], test_size=test_size, stratify=df["label"], random_state=seed)
