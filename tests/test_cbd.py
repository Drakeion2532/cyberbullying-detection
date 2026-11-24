# Copyright (c) 2026 Shubhan Punde · MIT licence
import pandas as pd
import pytest

from cbd.data import load
from cbd.preprocess import clean, normalise, tokens
from cbd.train import run


def test_clean_strips_urls_mentions_and_keeps_hashtag_words():
    t = clean("RT @user Check this https://t.co/abc #Muslims are 😡 &amp; 100% great")
    assert "http" not in t and "@" not in t and "user" not in t
    assert "muslims" in t and "100" not in t and "&" not in t


def test_negations_survive_preprocessing():
    assert "not" in tokens("You aren't stupid") and "not" in tokens("you are not stupid")
    assert normalise("you are not stupid") != normalise("you are stupid")


def test_duplicates_and_conflicting_labels_are_removed(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"tweet_text": ["a b c", "a b c", "x y", "x y", "other one"],
                  "cyberbullying_type": ["age", "age", "age", "gender", "not_cyberbullying"]}).to_csv(p, index=False)
    df = load("six_class", path=p)
    assert sorted(df["text"]) == ["a b c", "other one"]      # duplicate kept once; conflicting tweet dropped


def test_five_class_drops_other_cyberbullying(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"tweet_text": ["a", "b"], "cyberbullying_type": ["other_cyberbullying", "religion"]}).to_csv(p, index=False)
    assert load("five_class", path=p)["label"].tolist() == ["religion"]


def test_pipeline_learns_a_toy_task():
    words = {"age": "school kid teen", "gender": "girl woman gay", "religion": "muslim christian god",
             "ethnicity": "black white asian", "not_cyberbullying": "sunny lunch football"}
    rows = [(f"{w} {w.split()[i % 3]} message {i}", lab) for lab, w in words.items() for i in range(40)]
    df = pd.DataFrame(rows, columns=["text", "label"])
    out = run(df=df, fast=True, save=False)
    assert out["models"]["ensemble"]["accuracy"] > 0.9
    assert set(out["models"]) == {"logistic_regression", "svm", "random_forest", "xgboost", "lightgbm", "ensemble"}


@pytest.mark.parametrize("task", ["five_class", "six_class"])
def test_real_dataset_shape(task):
    df = load(task)
    assert df["text"].is_unique
    assert len(df) > 38000 if task == "five_class" else len(df) > 44000
