# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/cyberbullying-detection · MIT licence · sp2532-f30cd46f33af
"""Draw the README figures from results/metrics_<task>.json.

    python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"logistic_regression": "Logistic Reg.", "svm": "SVM", "random_forest": "Random Forest",
         "xgboost": "XGBoost", "lightgbm": "LightGBM", "ensemble": "Ensemble"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(task):
    p = ROOT / "results" / f"metrics_{task}.json"
    return json.loads(p.read_text()) if p.exists() else None


def model_comparison(m5, m6):
    keys = list(NAMES)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 3.9))
    for off, m, lab, c in [(-0.2, m5, "5 classes", "#2563eb"), (0.2, m6, "6 classes (incl. 'other')", "#94a3b8")]:
        if m is None:
            continue
        v = [m["models"][k]["accuracy"] * 100 for k in keys]
        bars = ax.bar(x + off, v, 0.38, label=lab, color=c)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, [NAMES[k] for k in keys])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(70, 100)
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=9)
    ax.set_title("Accuracy on the held-out 20% (this reimplementation)", loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=160)


def per_class_f1(m):
    rep = m["ensemble_report"]
    cls = m["classes"]
    f1 = [rep[c]["f1-score"] for c in cls]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.barh([cls[i].replace("_", " ") for i in order], [f1[i] for i in order], color="#2563eb")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("F1 score (ensemble)")
    ax.set_title(f"Per-class F1, {m['task'].replace('_', '-')} task", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / f"per_class_f1_{m['task']}.png", dpi=160)


def confusion(m):
    cm = np.array(m["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    short = {"not_cyberbullying": "not\nbullying", "other_cyberbullying": "other"}
    cls = [short.get(c, c) for c in m["classes"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(cls)), cls, fontsize=8)
    ax.set_yticks(range(len(cls)), cls, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Ensemble confusion matrix (row-normalised)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{m['task']}.png", dpi=160)


def pipeline_diagram():
    steps = ["Tweet", "Clean\n(URLs, @, emoji,\nnegations kept)", "TF-IDF 1-2 grams\n+ Word2Vec mean",
             "SMOTE\n(train only)", "LR · SVM · RF\nXGBoost · LightGBM", "Soft vote\n(avg. probability)", "Label +\nconfidence"]
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.axis("off")
    w, gap = 1.35, 0.25
    for i, s in enumerate(steps):
        x0 = i * (w + gap)
        ax.add_patch(plt.Rectangle((x0, 0), w, 1, fc="#eff6ff" if i not in (0, 6) else "#2563eb",
                                   ec="#2563eb", lw=1.2))
        ax.text(x0 + w / 2, 0.5, s, ha="center", va="center", fontsize=8.5,
                color="white" if i in (0, 6) else "#0f172a")
        if i < len(steps) - 1:
            ax.annotate("", (x0 + w + gap, 0.5), (x0 + w, 0.5), arrowprops=dict(arrowstyle="->", color="#2563eb"))
    ax.set_xlim(-0.05, len(steps) * (w + gap))
    ax.set_ylim(-0.1, 1.1)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=160)


if __name__ == "__main__":
    m5, m6 = load("five_class"), load("six_class")
    pipeline_diagram()
    model_comparison(m5, m6)
    for m in (m5, m6):
        if m:
            per_class_f1(m)
            confusion(m)
    print("figures ->", OUT)
