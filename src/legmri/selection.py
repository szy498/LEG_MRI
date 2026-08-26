"""Fold-local preprocessing, texture stability and L1 logistic feature selection."""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def correlation_keep(frame, threshold, priority=()):
    """Greedy absolute-Spearman pruning; declared clinical priority, then column order."""
    order = [c for c in priority if c in frame]
    order.extend(c for c in frame if c not in order)
    corr = frame[order].corr(method="spearman").abs()
    kept = []
    for col in order:
        if not any(corr.loc[col, previous] > threshold for previous in kept):
            kept.append(col)
    return kept


class StudySelector(TransformerMixin, BaseEstimator):
    """All learned quantities are estimated only from the data passed to fit.

    For combined imaging, per-domain selectors are fitted inside training-only
    subfolds. Candidates with frequency strictly greater than 80% are pooled,
    re-pruned and L1-selected. Clinical covariates are always retained.
    """

    def __init__(
        self,
        quantitative=(),
        radiomics=(),
        clinical=(),
        settings=None,
        lasso_c=1.0,
        random_state=42,
    ):
        self.quantitative = quantitative
        self.radiomics = radiomics
        self.clinical = clinical
        self.settings = settings
        self.lasso_c = lasso_c
        self.random_state = random_state

    def _prepare(self, frame):
        frame = frame.replace([np.inf, -np.inf], np.nan)
        self.medians_ = frame.median().fillna(0)
        filled = frame.fillna(self.medians_)
        self.means_ = filled.mean()
        self.scales_ = filled.std(ddof=0).replace(0, 1)
        return filled, (filled - self.means_) / self.scales_

    def _single_domain(self, frame, y, columns):
        cfg = self.settings or {}
        suffixes = tuple(cfg.get("excluded_suffixes", []))
        columns = [c for c in columns if not (suffixes and c.endswith(suffixes))]
        if not columns:
            return []
        raw = frame[columns].replace([np.inf, -np.inf], np.nan)
        raw = raw.fillna(raw.median()).fillna(0)
        columns = list(raw.columns[raw.nunique() > 1])
        if not columns:
            return []
        raw = raw[columns]
        threshold, priority = cfg.get("correlation_threshold", 0.9), cfg.get("priority", [])
        retained = correlation_keep(raw, threshold, priority)
        textures = [c for c in retained if "_glcm_" in c]
        if textures:
            count = Counter()
            n = cfg.get("texture_stability_splits", 10)
            splits = StratifiedShuffleSplit(
                n_splits=n,
                random_state=self.random_state,
                train_size=cfg.get("texture_train_fraction", 0.7),
            )
            for train, _ in splits.split(raw, y):
                count.update(correlation_keep(raw.iloc[train], threshold, priority))
            retained = [
                c
                for c in retained
                if c not in textures or count[c] / n > cfg.get("texture_frequency", 0.8)
            ]
        return self._lasso(raw, y, retained)

    def _lasso(self, raw, y, columns):
        if not columns:
            return []
        x = raw[columns]
        x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median()).fillna(0)
        x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=self.lasso_c,
            class_weight="balanced",
            max_iter=5000,
            random_state=self.random_state,
        )
        model.fit(x, y)
        return [c for c, coefficient in zip(columns, model.coef_[0]) if abs(coefficient) > 1e-8]

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StudySelector requires named DataFrame columns")
        y = np.asarray(y)
        self.feature_names_in_ = np.asarray(X.columns, object)
        self.n_features_in_ = X.shape[1]
        cfg = self.settings or {}
        q, r = list(self.quantitative), list(self.radiomics)
        self.integration_frequencies_ = {}
        if q and r:
            folds = cfg.get("integration_folds", 5)
            splitter = StratifiedKFold(folds, shuffle=True, random_state=self.random_state)
            counts = Counter()
            for train, _ in splitter.split(X, y):
                for domain in (q, r):
                    counts.update(self._single_domain(X.iloc[train], y[train], domain))
            self.integration_frequencies_ = {c: counts[c] / folds for c in q + r}
            candidates = [
                c for c in q + r if counts[c] / folds > cfg.get("integration_frequency", 0.8)
            ]
            raw = X[candidates].replace([np.inf, -np.inf], np.nan)
            raw = raw.fillna(raw.median()).fillna(0)
            candidates = correlation_keep(
                raw, cfg.get("correlation_threshold", 0.9), cfg.get("priority", [])
            )
            chosen = self._lasso(raw, y, candidates)
        else:
            chosen = self._single_domain(X, y, q or r)
        self.selected_features_ = chosen + list(self.clinical)
        self._prepare(X[self.selected_features_])
        # An intercept-only model is a valid result, not grounds for dropping a fold.
        self.output_features_ = self.selected_features_ or ["__intercept_only__"]
        return self

    def transform(self, X):
        if not self.selected_features_:
            return np.zeros((len(X), 1), dtype=float)
        x = X[self.selected_features_].replace([np.inf, -np.inf], np.nan)
        return ((x.fillna(self.medians_) - self.means_) / self.scales_).to_numpy(float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_features_, object)
