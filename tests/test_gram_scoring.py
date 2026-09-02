"""Gram-based validation scoring must reproduce row-based scoring exactly.

The fast gaussian CV path can score a learner on the validation rows of a
holdout either by materializing predictions (row path) or from precomputed
unweighted validation Gram matrices (gram path, default). The two are the
same algebra in a different floating point order, so scores must agree to
~1e-10 -- including with non-uniform weights/trim_weights, because the
holdout score (globals.get_rmse) is itself unweighted.
"""

import numpy as np
import pandas as pd
import pytest

from modrover.learner import ModelStatus
from modrover.rover import Rover, _gram_scoring_enabled


def _make_data(seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    n = 300
    covs = [f"x{i}" for i in range(6)]
    df = pd.DataFrame(rng.normal(size=(n, len(covs))), columns=covs)
    df["intercept"] = 1.0
    beta = np.array([1.5, -2.0, 0.0, 0.5, 0.0, 1.0])
    df["y"] = df[covs].to_numpy().dot(beta) + 0.3 + rng.normal(0, 0.5, size=n)
    # Non-uniform weights: the train-side solves are weighted, but the
    # holdout score is not -- gram scoring must be exact regardless.
    df["weights"] = rng.uniform(0.5, 2.0, size=n)
    df["trim_weights"] = rng.uniform(0.8, 1.0, size=n)
    for h in range(3):
        df[f"holdout_{h}"] = (rng.uniform(size=n) < 0.3).astype(int)
    return df, covs


def _fit_rover(df: pd.DataFrame, covs: list[str], gram: bool, monkeypatch) -> Rover:
    monkeypatch.setenv("MODROVER_GRAM_SCORING", "1" if gram else "0")
    rover = Rover(
        model_type="gaussian",
        obs="y",
        cov_fixed=["intercept"],
        cov_exploring=covs,
        weights="weights",
        holdouts=[f"holdout_{h}" for h in range(3)],
    )
    rover._explore(
        data=df,
        strategies=["forward"],
        strategy_options={"forward": {"max_len": len(covs)}},
    )
    return rover


def test_env_var_toggles_gram_cache(monkeypatch):
    monkeypatch.setenv("MODROVER_GRAM_SCORING", "0")
    assert not _gram_scoring_enabled()
    monkeypatch.setenv("MODROVER_GRAM_SCORING", "1")
    assert _gram_scoring_enabled()
    monkeypatch.delenv("MODROVER_GRAM_SCORING", raising=False)
    assert _gram_scoring_enabled()  # default ON

    df, covs = _make_data()
    rover_on = _fit_rover(df, covs, gram=True, monkeypatch=monkeypatch)
    holdout_data, _ = rover_on._prepare_holdout_data(
        df, [f"holdout_{h}" for h in range(3)]
    )
    assert all("val_xtx" in hd[3] for hd in holdout_data.values())

    monkeypatch.setenv("MODROVER_GRAM_SCORING", "0")
    holdout_data, _ = rover_on._prepare_holdout_data(
        df, [f"holdout_{h}" for h in range(3)]
    )
    assert all("val_xtx" not in hd[3] for hd in holdout_data.values())


def test_gram_matches_row_scoring(monkeypatch):
    df, covs = _make_data()
    rover_row = _fit_rover(df, covs, gram=False, monkeypatch=monkeypatch)
    rover_gram = _fit_rover(df, covs, gram=True, monkeypatch=monkeypatch)

    assert set(rover_row.learners) == set(rover_gram.learners)
    assert len(rover_row.learners) > 10

    max_delta = 0.0
    n_success = 0
    for lid, lrn_row in rover_row.learners.items():
        lrn_gram = rover_gram.learners[lid]
        assert lrn_row.status == lrn_gram.status
        if lrn_row.status != ModelStatus.SUCCESS:
            continue
        n_success += 1
        # overall score (mean of per-holdout cv scores)
        max_delta = max(max_delta, abs(lrn_row.score - lrn_gram.score))
        # per-holdout cv scores
        assert set(lrn_row._cv_scores) == set(lrn_gram._cv_scores)
        for holdout, s_row in lrn_row._cv_scores.items():
            max_delta = max(
                max_delta, abs(s_row - lrn_gram._cv_scores[holdout])
            )
        # final coefficients must be untouched by the scoring path
        np.testing.assert_allclose(
            lrn_row.coef, lrn_gram.coef, rtol=0, atol=0
        )
    assert n_success > 10
    assert max_delta < 1e-10


def test_gram_score_against_manual_rmse(monkeypatch):
    """Gram cv score equals exp(-sqrt(mean((obs - X_v c)^2))) recomputed
    by hand from the learner's cv coefficients."""
    df, covs = _make_data(seed=7)
    holdouts = [f"holdout_{h}" for h in range(3)]
    rover = _fit_rover(df, covs, gram=True, monkeypatch=monkeypatch)

    lid = (0, 1, 3)
    learner = rover.learners[lid]
    assert learner.status == ModelStatus.SUCCESS

    # re-derive the per-holdout coefficient exactly as the fast path does
    cov_names = ["intercept"] + [covs[i] for i in lid]
    for holdout in holdouts:
        train = df[df[holdout] == 0]
        val = df[df[holdout] == 1]
        x = train[cov_names].to_numpy(dtype=float)
        y = train["y"].to_numpy(dtype=float)
        w = (
            train["weights"].to_numpy(dtype=float)
            * train["trim_weights"].to_numpy(dtype=float)
        )
        sqrt_w = np.sqrt(w)
        xw = x * sqrt_w[:, None]
        xtwx = xw.T.dot(xw)
        xtwy = xw.T.dot(y * sqrt_w)
        coef = np.linalg.solve(xtwx, xtwy)
        pred = val[cov_names].to_numpy(dtype=float).dot(coef)
        obs = val["y"].to_numpy(dtype=float)
        expected = float(np.exp(-np.sqrt(np.mean((obs - pred) ** 2))))
        assert learner._cv_scores[holdout] == pytest.approx(
            expected, abs=1e-10
        )
