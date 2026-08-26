"""Optional XGBoost adapter; class balance is computed within each fit."""

import numpy as np
from xgboost import XGBClassifier


class FoldBalancedXGB(XGBClassifier):
    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.set_params(scale_pos_weight=float(np.sum(y == 0) / np.sum(y == 1)))
        return super().fit(X, y, **kwargs)
