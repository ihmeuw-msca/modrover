"""Batched layer solves must reproduce the per-learner fitting path.

With MODROVER_BATCHED_SOLVES=1 (default) the fast gaussian CV path fits all
children of an exploration layer through one stacked np.linalg.solve per
(holdout, full) fit plus vectorized gram scoring, and Learners share cached
regmod Variable objects. The stacked gesv factors exactly the same matrices
as the per-learner solves (coefficients are expected bit-identical); only
the score/vcov reductions may reorder floating point, so those must agree
to ~1e-12. Statuses -- including SINGULAR children inside an otherwise
healthy layer -- must be identical.
"""

import numpy as np
import pandas as pd
import pytest

from modrover.learner import ModelStatus
from modrover.rover import Rover, _batched_solves_enabled


def _make_data(seed: int = 42, singular: bool = False):
    rng = np.random.default_rng(seed)
    n = 300
    covs = [f"x{i}" for i in range(6)]
    df = pd.DataFrame(rng.normal(size=(n, len(covs))), columns=covs)
    if singular:
        # exact duplicate column: every learner containing both x0 and
        # x_dup has a singular Gram matrix
        df["x_dup"] = df["x0"]
        covs = covs + ["x_dup"]
    df["intercept"] = 1.0
    beta = np.array([1.5, -2.0, 0.0, 0.5, 0.0, 1.0])
    df["y"] = (
        df[covs[:6]].to_numpy().dot(beta) + 0.3 + rng.normal(0, 0.5, size=n)
    )
    df["weights"] = rng.uniform(0.5, 2.0, size=n)
    df["trim_weights"] = rng.uniform(0.8, 1.0, size=n)
    for h in range(3):
        df[f"holdout_{h}"] = (rng.uniform(size=n) < 0.3).astype(int)
    return df, covs


def _fit_rover(df, covs, batched: bool, monkeypatch, max_len: int = 3) -> Rover:
    monkeypatch.setenv("MODROVER_BATCHED_SOLVES", "1" if batched else "0")
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
        strategy_options={"forward": {"max_len": max_len}},
    )
    return rover


def test_env_var_toggles_batched_solves(monkeypatch):
    monkeypatch.setenv("MODROVER_BATCHED_SOLVES", "0")
    assert not _batched_solves_enabled()
    monkeypatch.setenv("MODROVER_BATCHED_SOLVES", "1")
    assert _batched_solves_enabled()
    monkeypatch.delenv("MODROVER_BATCHED_SOLVES", raising=False)
    assert _batched_solves_enabled()  # default ON


def _assert_paths_identical(rover_b: Rover, rover_p: Rover):
    # identical learner sets in identical insertion order (learner_info /
    # learner-table row order depends on it)
    assert list(rover_b.learners) == list(rover_p.learners)

    max_dscore, n_success = 0.0, 0
    for lid, lrn_b in rover_b.learners.items():
        lrn_p = rover_p.learners[lid]
        assert lrn_b.status == lrn_p.status
        assert lrn_b._cv_status == lrn_p._cv_status
        if lrn_b.status != ModelStatus.SUCCESS:
            continue
        n_success += 1
        max_dscore = max(max_dscore, abs(lrn_b.score - lrn_p.score))
        assert set(lrn_b._cv_scores) == set(lrn_p._cv_scores)
        for holdout, s_p in lrn_p._cv_scores.items():
            max_dscore = max(
                max_dscore, abs(s_p - lrn_b._cv_scores[holdout])
            )
        # stacked gesv on the same matrices: coefficients bit-identical
        np.testing.assert_array_equal(lrn_b.coef, lrn_p.coef)
        # vcov differs only through the einsum-reordered sigma2 scalar
        np.testing.assert_allclose(
            lrn_b.vcov, lrn_p.vcov, rtol=1e-12, atol=1e-15
        )
    assert n_success > 10
    assert max_dscore < 1e-12
    return max_dscore


def test_batched_matches_per_learner(monkeypatch):
    df, covs = _make_data()
    rover_b = _fit_rover(df, covs, batched=True, monkeypatch=monkeypatch)
    rover_p = _fit_rover(df, covs, batched=False, monkeypatch=monkeypatch)
    assert len(rover_b.learners) > 10
    _assert_paths_identical(rover_b, rover_p)


def test_batched_with_singular_children(monkeypatch):
    """A deliberately singular child (duplicated column) inside a layer must
    be marked exactly as the per-learner path marks it, without disturbing
    its siblings."""
    df, covs = _make_data(seed=3, singular=True)
    rover_b = _fit_rover(
        df, covs, batched=True, monkeypatch=monkeypatch, max_len=len(covs)
    )
    rover_p = _fit_rover(
        df, covs, batched=False, monkeypatch=monkeypatch, max_len=len(covs)
    )
    statuses = {l.status for l in rover_b.learners.values()}
    assert ModelStatus.SUCCESS in statuses
    # the duplicated-column learners fail CV (SINGULAR on their first
    # holdout) in both paths
    assert any(
        l.status != ModelStatus.SUCCESS for l in rover_b.learners.values()
    )
    _assert_paths_identical(rover_b, rover_p)
    # spot-check: any learner containing both duplicated columns is failed
    # and carries a SINGULAR cv status
    i_dup = covs.index("x_dup")
    for lid, lrn in rover_b.learners.items():
        if 0 in lid and i_dup in lid:
            assert lrn.status == ModelStatus.CV_FAILED
            assert ModelStatus.SINGULAR in lrn._cv_status.values()


def test_variable_cache_shared_across_learners(monkeypatch):
    """On the fast gaussian path, Variables are constructed once per Rover
    and shared across learners (they are read-only there); the per-learner
    spec dicts and variable lists remain fresh objects."""
    df, covs = _make_data()
    rover = _fit_rover(df, covs, batched=True, monkeypatch=monkeypatch)
    l1 = rover._get_learner((0,), use_cache=False)
    l2 = rover._get_learner((0, 1), use_cache=False)
    v1 = l1.param_specs["mu"]["variables"]
    v2 = l2.param_specs["mu"]["variables"]
    assert v1[0] is v2[0]  # intercept shared
    assert v1[1] is v2[1]  # x0 shared
    assert v1 is not v2  # but the lists are per-learner
    assert l1.param_specs["mu"] is not l2.param_specs["mu"]
    assert [v.name for v in v1] == ["intercept", "x0"]
    assert [v.name for v in v2] == ["intercept", "x0", "x1"]
    assert len(rover._variable_cache) == len(covs) + 1
    # cached Variables carry no priors and are never mutated by the fast path
    for var in rover._variable_cache.values():
        assert var.priors == []

    # with batching disabled, the deepcopy path constructs fresh Variables
    monkeypatch.setenv("MODROVER_BATCHED_SOLVES", "0")
    l3 = rover._get_learner((0,), use_cache=False)
    assert l3.param_specs["mu"]["variables"][0] is not v1[0]


def test_batched_solve_matches_manual_solution(monkeypatch):
    """Coefficients from the batched path equal the directly computed
    weighted least-squares solution."""
    df, covs = _make_data(seed=7)
    rover = _fit_rover(df, covs, batched=True, monkeypatch=monkeypatch)
    lid = (0, 1, 3)
    learner = rover.learners[lid]
    assert learner.status == ModelStatus.SUCCESS
    names = ["intercept"] + [covs[i] for i in lid]
    x = df[names].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    w = (
        df["weights"].to_numpy(dtype=float)
        * df["trim_weights"].to_numpy(dtype=float)
    )
    xw = x * np.sqrt(w)[:, None]
    coef = np.linalg.solve(xw.T @ xw, xw.T @ (y * np.sqrt(w)))
    assert learner.coef == pytest.approx(coef, abs=1e-10)
