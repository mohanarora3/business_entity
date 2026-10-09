"""Matching classifier.

LightGBM (MIT license) is the intended model. If LightGBM is not installed, scikit-learn's
HistGradientBoostingClassifier (same algorithm family) is used so the pipeline still runs.
Both are small tabular models — far under the 8B-parameter / MIT-Apache constraint.
"""

import numpy as np

import config as C

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False


def make_model(n_pos: int, n_neg: int):
    if HAVE_LGB:
        return lgb.LGBMClassifier(
            n_estimators=700,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=C.SEED,
            n_jobs=-1,
            verbose=-1,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=C.SEED,
    )


def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def name() -> str:
    return "LightGBM" if HAVE_LGB else "sklearn HistGradientBoosting (fallback)"
