# Copyright (c) 2026 Shubhan Punde · MIT licence
"""Train the five models and the soft-voting ensemble, evaluate on a held-out 20%.

    python -m cbd.train                 # five-class task (default)
    python -m cbd.train --task six_class
    python -m cbd.train --fast          # smaller forests for a quick check

The features are fitted once on the training split and shared by every model, so the
model comparison is fair, and SMOTE runs on the training features only.
Outputs: results/metrics_<task>.json and models/ensemble_<task>.joblib
(the fitted featuriser + the five models, ready for cbd.predict and the Streamlit app).
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from . import MODELS, RESULTS
from .data import load, split
from .features import TextFeatures
from .models import SEED, SoftVote, base_models


def run(task="five_class", fast=False, df=None, save=True):
    df = load(task) if df is None else df
    Xtr, Xte, ytr, yte = split(df)
    enc = LabelEncoder().fit(df["label"])
    ytr_i, yte_i = enc.transform(ytr), enc.transform(yte)

    t0 = time.time()
    feats = TextFeatures().fit(Xtr)
    Ftr, Fte = feats.transform(Xtr), feats.transform(Xte)
    Ftr_s, ytr_s = SMOTE(random_state=SEED).fit_resample(Ftr, ytr_i)
    timing = {"features_s": round(time.time() - t0, 1)}

    results, fitted = {}, {}
    for name, model in base_models(fast).items():
        t = time.time()
        model.fit(Ftr_s, ytr_s)
        fitted[name] = model
        pred = model.predict_proba(Fte).argmax(1)
        results[name] = {"accuracy": accuracy_score(yte_i, pred), "macro_f1": f1_score(yte_i, pred, average="macro"),
                         "train_s": round(time.time() - t, 1)}
        print(f"{name:20s} acc {results[name]['accuracy']:.4f}  macro-F1 {results[name]['macro_f1']:.4f}")

    vote = SoftVote(feats, fitted, enc.classes_.tolist())
    pred = vote.proba_from_features(Fte).argmax(1)
    report = classification_report(yte_i, pred, target_names=enc.classes_, output_dict=True)
    results["ensemble"] = {"accuracy": accuracy_score(yte_i, pred), "macro_f1": f1_score(yte_i, pred, average="macro")}
    print(f"{'ensemble':20s} acc {results['ensemble']['accuracy']:.4f}  macro-F1 {results['ensemble']['macro_f1']:.4f}")

    out = {"task": task, "n_tweets": int(len(df)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
           "classes": enc.classes_.tolist(), "models": results, "ensemble_report": report,
           "confusion_matrix": confusion_matrix(yte_i, pred).tolist(), "timing": timing}
    if save:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"metrics_{task}.json").write_text(json.dumps(out, indent=2))
        MODELS.mkdir(exist_ok=True)
        joblib.dump(vote, MODELS / f"ensemble_{task}.joblib", compress=3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="five_class", choices=["five_class", "six_class"])
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    run(a.task, a.fast)


if __name__ == "__main__":
    main()
