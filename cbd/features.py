# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/cyberbullying-detection · MIT licence · sp2532-f30cd46f33af
"""TF-IDF + Word2Vec features.

TF-IDF (word 1-2 grams on cleaned, stemmed text) captures which words and phrases are
used. Word2Vec, trained on the training tweets only, adds the average meaning of a tweet,
so related insults that never co-occur ("dumb", "idiot") still land close together.
The two are stacked side by side into one feature matrix.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .preprocess import normalise, tokens


class Word2VecMean(BaseEstimator, TransformerMixin):
    def __init__(self, dim=100, window=5, min_count=2, epochs=10, seed=42):
        self.dim, self.window, self.min_count, self.epochs, self.seed = dim, window, min_count, epochs, seed

    def fit(self, X, y=None):
        from gensim.models import Word2Vec
        docs = [tokens(t) for t in X]
        self.w2v_ = Word2Vec(docs, vector_size=self.dim, window=self.window, min_count=self.min_count,
                             epochs=self.epochs, seed=self.seed, workers=1)
        return self

    def transform(self, X):
        kv = self.w2v_.wv
        out = np.zeros((len(X), self.dim), dtype=np.float32)
        for i, t in enumerate(X):
            vecs = [kv[w] for w in tokens(t) if w in kv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return normalize(out)


class TextFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000, ngram_range=(1, 2), w2v_dim=100):
        self.max_features, self.ngram_range, self.w2v_dim = max_features, ngram_range, w2v_dim

    def fit(self, X, y=None):
        X = list(X)
        self.tfidf_ = TfidfVectorizer(preprocessor=normalise, token_pattern=r"(?u)\b\w+\b",
                                      ngram_range=self.ngram_range, max_features=self.max_features,
                                      min_df=2, sublinear_tf=True).fit(X)
        self.w2v_ = Word2VecMean(dim=self.w2v_dim).fit(X)
        return self

    def transform(self, X):
        X = list(X)
        return sparse.hstack([self.tfidf_.transform(X), sparse.csr_matrix(self.w2v_.transform(X))]).tocsr()
