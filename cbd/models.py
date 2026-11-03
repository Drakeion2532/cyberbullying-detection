# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/cyberbullying-detection · MIT licence · sp2532-f30cd46f33af
"""The five models and the soft-voting ensemble described in the paper.

LightGBM, XGBoost, SVM, Random Forest and Logistic Regression each see the same
TF-IDF + Word2Vec features. SMOTE oversamples the smaller classes inside the training
pipeline only, so synthetic samples never leak into the test set. The ensemble averages
the five models' class probabilities (soft voting).
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .features import TextFeatures

SEED = 42


def base_models(fast: bool = False) -> dict:
    n = 60 if fast else 300
    return {
        "logistic_regression": LogisticRegression(C=4.0, max_iter=2000),
        "svm": CalibratedClassifierCV(LinearSVC(C=0.5), cv=3),          # calibrated so it can vote with probabilities
        "random_forest": RandomForestClassifier(n_estimators=n, min_samples_leaf=1, n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.15, tree_method="hist",
                                 subsample=0.9, colsample_bytree=0.5, n_jobs=-1, random_state=SEED, verbosity=0),
        "lightgbm": LGBMClassifier(n_estimators=n, num_leaves=63, learning_rate=0.1, colsample_bytree=0.5,
                                   random_state=SEED, n_jobs=-1, verbose=-1),
    }


def pipeline(estimator, max_features=20000) -> Pipeline:
    return Pipeline([("features", TextFeatures(max_features=max_features)),
                     ("smote", SMOTE(random_state=SEED)),
                     ("model", estimator)])


def ensemble(fast: bool = False) -> VotingClassifier:
    """The same ensemble as a scikit-learn VotingClassifier (for use on precomputed features)."""
    return VotingClassifier(list(base_models(fast).items()), voting="soft", n_jobs=1)


class SoftVote:
    """Fitted featuriser + fitted models; averages the models' class probabilities.

    Equivalent to VotingClassifier(voting="soft") but reuses models that were already
    trained during evaluation, so training runs once.
    """

    def __init__(self, features, models: dict, classes: list[str]):
        self.features, self.models, self.classes = features, models, classes

    def proba_from_features(self, F):
        import numpy as np
        return np.mean([m.predict_proba(F) for m in self.models.values()], axis=0)

    def predict_proba(self, texts):
        return self.proba_from_features(self.features.transform(list(texts)))

    def predict(self, texts):
        return [self.classes[i] for i in self.predict_proba(texts).argmax(1)]

    def explain(self, text: str) -> dict:
        """Per-model vote for one text, for the app and for debugging."""
        F = self.features.transform([text])
        return {name: self.classes[int(m.predict_proba(F).argmax(1)[0])] for name, m in self.models.items()}
