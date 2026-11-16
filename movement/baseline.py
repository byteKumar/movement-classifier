"""The gradient boosting baseline recipe, shared by every script that runs it."""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

PARAMS = dict(learning_rate=0.1, max_iter=500, max_leaf_nodes=31, l2_regularization=0.0, early_stopping=False)


def _at_round(model, F, n_round: int) -> np.ndarray:
    for i, p in enumerate(model.staged_predict(F), start=1):
        if i == n_round:
            return p
    raise ValueError(n_round)


def fit_and_predict(F_tr, y_tr, F_va, y_va, F_eval_list, seed: int) -> dict:
    """Fit 500 rounds, pick the round with the best val macro-F1, predict at that round.

    Returns {"best_iter", "fit_seconds", "val_pred", "preds": [one array per F_eval_list item]}.
    """
    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(**PARAMS, random_state=seed).fit(F_tr, y_tr)
    fit_s = time.perf_counter() - t0
    curve = [f1_score(y_va, p, average="macro") for p in model.staged_predict(F_va)]
    best = int(np.argmax(curve)) + 1  # first round reaching the best val score
    return {"best_iter": best, "fit_seconds": fit_s, "val_pred": _at_round(model, F_va, best),
            "preds": [_at_round(model, F, best) for F in F_eval_list]}
