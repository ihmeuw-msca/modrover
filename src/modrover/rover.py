import os
from copy import deepcopy
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.variable import Variable

from .exceptions import InvalidConfigurationError, NotFittedError
from .globals import get_rmse, model_type_dict
from .learner import Learner, LearnerID, ModelStatus
from .strategies import get_strategy


def _gram_scoring_enabled() -> bool:
    """Gram-based validation scoring is ON by default; set the environment
    variable ``MODROVER_GRAM_SCORING=0`` to fall back to row-based scoring."""
    return os.environ.get("MODROVER_GRAM_SCORING", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def _batched_solves_enabled() -> bool:
    """Batched per-layer solves (and the shared-Variable cache) are ON by
    default; set ``MODROVER_BATCHED_SOLVES=0`` to fall back to the
    per-learner fitting path."""
    return os.environ.get(
        "MODROVER_BATCHED_SOLVES", "1"
    ).strip().lower() not in ("0", "false", "off")


# children per batched linear solve; bounds the (C, s, s) stack memory
_BATCH_CHUNK = 2048


class Rover:
    """Rover class explores model space and creates final super learner for
    prediction and inference.

    Parameters
    ----------
    model_type
        Type of the model. For example ``"gaussian"`` or ``"poisson"``
    obs
        The name of the column representing observations
    cov_fixed
        A list representing the covariates are present in every learner
    cov_exploring
        A list representing the covariates rover will explore over
    main_param
        The main parameter where the ``cov_fixed`` and ``cov_exploring`` are
        applied to. By default ``main_param=None``, and when the model only have
        one parameter, ``main_param`` will be automatically re-assigned to be
        that parameter. If we have multiple parameters in the model, user has to
        specify ``main_param``.
    param_specs
        Parameter settings including, link function, priors, etc
    weights
        Column name corresponding to the weights for each data point
    holdouts
        A list of column names containing 1's and 0's that represent folds in
        the rover cross-validation algorithm
    get_score
        A callable used to evaluate cross-validated score of sub-learners
        in rover

    """

    def __init__(
        self,
        model_type: str,
        obs: str,
        cov_fixed: list[str],
        cov_exploring: list[str],
        main_param: str | None = None,
        param_specs: dict[str, dict] | None = None,
        weights: str = "weights",
        holdouts: list[str] | None = None,
        get_score: Callable = get_rmse,
    ) -> None:
        self.model_type = self._as_model_type(model_type)
        self.obs = obs
        self.cov_fixed, self.cov_exploring = self._as_cov(
            cov_fixed, cov_exploring
        )
        self.main_param = self._as_main_param(main_param)
        self.param_specs = self._as_param_specs(param_specs)
        self.weights = weights
        self.holdouts = holdouts
        self.get_score = get_score

        self.learners: dict[LearnerID, Learner] = {}
        self._coef_index_cache: dict[LearnerID, list[int]] = {}
        self._variable_cache: dict[str, Variable] = {}

    @property
    def model_class(self) -> type:
        """Model class that ``model_type`` refers to."""
        return model_type_dict[self.model_type]

    @property
    def params(self) -> tuple[str, ...]:
        """A tuple of parameter names belong to the model class."""
        return self.model_class.param_names

    @property
    def variables(self) -> tuple[str, ...]:
        """A tuple of the variable names belong the model class with full list
        of covariates.
        """
        names = []
        for p in self.params:
            names.extend([f"{p}_{v}" for v in self.param_specs[p]["variables"]])
            if p == self.main_param:
                names.extend([f"{p}_{v}" for v in self.cov_exploring])
        return tuple(names)

    @property
    def num_vars(self) -> int:
        """Number of variables with full list of covariates."""
        return len(self.variables)

    @property
    def super_learner_id(self) -> tuple[int, ...]:
        """Learner id for the super learner."""
        return tuple(range(len(self.cov_exploring)))

    @property
    def super_learner(self) -> Learner:
        """Ensembled super learner."""
        if not hasattr(self, "_super_learner"):
            raise NotFittedError("Rover has not been ensembled yet")
        return self._super_learner

    @property
    def learner_info(self) -> DataFrame:
        """A data frame contains important information of fitted learners."""
        if not hasattr(self, "_learner_info"):
            raise NotFittedError("Rover has not been ensemble yet")
        return self._learner_info

    @property
    def summary(self) -> DataFrame:
        """A data frame contains the summary information of explored covariates
        across all fitted learners.
        """
        if not hasattr(self, "_summary"):
            raise NotFittedError("Rover has not been ensemble yet")
        return self._summary

    def fit(
        self,
        data: DataFrame,
        strategies: list[str],
        strategy_options: dict | None = None,
        top_pct_score: float = 0.1,
        top_pct_learner: float = 1.0,
        coef_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Fits the ensembled super learner.

        Explores over all covariate slices as defined by the input strategy, and
        fits the sublearners.

        The super learner coefficients are determined by the ensemble method
        parameters, and the super learner itself will be created - to be used in
        prediction and summarization.

        Parameters
        ----------
        data
            Training data to fit individual learners on.
        strategies
            The selection strategy to determine the model tree. Valid strategies
            include "forward", "backward" and "full".
        strategy_options
            A dictionary with key as the strategy name and value as the option
            with calling the strategy. By default, `strategy_options=None` where
            all default options will be used by the strategies.
        top_pct_score
            Only the learners with score that are greater or equal than
            ``best_score * (1 - top_score)`` can be selected. When
            ``top_score = 0`` only the best model will be selected.
        top_pct_learner
            Only the best ``top_pct_learner * num_learners`` will be selected.
        coef_bounds
            User pre-specified bounds for the coefficients. This is a dictionary
            with key as the covariate name and the value as the bounds. The
            learner will be marked valid or not if the coefficients are within
            the bounds. Invalid learners will not be used in ensemble process
            to create super learner. By default, ``coef_bounds=None``, where
            there is not validation based on the value of the coefficients.

        """
        self._explore(
            data=data, strategies=strategies, strategy_options=strategy_options
        )
        self._get_super_learner(
            top_pct_score=top_pct_score,
            top_pct_learner=top_pct_learner,
            coef_bounds=coef_bounds,
        )

    def predict(
        self, data: DataFrame, return_ui: bool = False, alpha: float = 0.05
    ) -> NDArray:
        """Predict with ensembled super learner.

        Parameters
        ----------
        data
            Testing data to predict
        return_ui
            If ``return_ui=True``, a matrix will be returned. The first row
            is the point prediction, second and thrid rows are the lower and
            upper bound of the prediction.
        alpha
            When ``return_ui=True``, function will return (1 - ``alpha``)
            uncertainty interval. By default, ``alpha=0.05``.

        """
        return self.super_learner.predict(
            data, return_ui=return_ui, alpha=alpha
        )

    def plot(self, bins: int | None = None) -> plt.Figure:
        """Plot the result of the exploration. Each panel of the figure
        corresponding to one covariate in the ``cov_exploring``. We plot the
        spread of the coefficients across all learners along with color
        represents their performance score.

        Parameters
        ----------
        bins
            When ``bins=None``, the coefficients will be spread along the y axis
            randomly to display the spread. When user pass in an integer, the
            x axis will be divided into bins and the y value will be assigned
            according to the ranking of score within the bin.

        """
        nrow = len(self.cov_exploring)
        fig, ax = plt.subplots(nrow, 1, figsize=(8, 2 * nrow), sharex=True)
        ax = [ax] if isinstance(ax, plt.Axes) else ax

        learner_info = self.learner_info[
            self.learner_info["status"] == ModelStatus.SUCCESS
        ].copy()
        if bins is not None:
            all_coef = learner_info[
                [f"{self.main_param}_{cov}" for cov in self.cov_exploring]
            ].to_numpy()
            cmin, cmax = all_coef.min(), all_coef.max()
            bins = np.linspace(cmin, cmax, bins + 1)

        summary = self.summary
        score_scaled = learner_info["score_scaled"].to_numpy()
        vmin, vmax = score_scaled.min(), score_scaled.max()
        highlight_index = {
            "final": learner_info["weight"] > 0,
            "invalid": ~learner_info["valid"],
        }
        highlight_config = {
            "single": {
                "marker": "^",
                "facecolor": "none",
                "edgecolor": "gray",
                "alpha": 0.5,
            },
            "final": {
                "marker": "o",
                "facecolor": "none",
                "edgecolor": "gray",
                "alpha": 0.5,
            },
            "invalid": {"marker": "x", "color": "gray", "alpha": 0.5},
        }
        for i, cov in enumerate(summary.sort_values("ranking")["cov"]):
            # plot the spread of the coef
            name = f"{self.main_param}_{cov}"
            coef = learner_info[name].to_numpy()
            coef_jitter = np.random.rand(coef.size)
            if bins is not None:
                learner_info["bin_id"] = np.digitize(learner_info[name], bins)
                coef_jitter = learner_info.groupby("bin_id")[
                    "score"
                ].rank() / len(learner_info)
            im = ax[i].scatter(
                coef,
                coef_jitter,
                alpha=0.2,
                c=score_scaled,
                edgecolors="none",
                vmin=vmin,
                vmax=vmax,
            )
            # mark single, final and invalid models
            highlight_index["single"] = learner_info["learner_id"] == (
                self.cov_exploring.index(cov),
            )
            for key, index in highlight_index.items():
                ax[i].scatter(
                    coef[index], coef_jitter[index], **highlight_config[key]
                )
            # indicator of 0
            ax[i].axvline(0, linewidth=1, color="gray", linestyle="--")
            # colorbar
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.ax.set_yticks([vmin, vmax])
            # plot ensemble result
            index = summary["cov"] == cov
            super_coef = summary[index]["coef"].iloc[0]
            super_coef_lwr = summary[index]["coef_lwr"].iloc[0]
            super_coef_upr = summary[index]["coef_upr"].iloc[0]
            ax[i].axvline(super_coef, linewidth=1, color="#008080")
            ax[i].plot(
                [super_coef_lwr, super_coef_upr],
                [0.5, 0.5],
                linewidth=1,
                color="#008080",
            )
            # summary text
            text = [
                f"ranking = {summary[index]['ranking'].iloc[0]} / {len(summary)}",
                f"significant = {summary[index]['significant'].iloc[0]}",
                f"pct_present = {summary[index]['pct_present'].iloc[0]:.2%}",
                f"score_improvement = {summary[index]['score_improvement'].iloc[0]:.4}",
                f"coef = {super_coef:.2f} ({super_coef_lwr:.2f}, {super_coef_upr:.2f})",
            ]
            text = "\n".join(text)
            ax[i].text(
                1.15,
                1,
                text,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[i].transAxes,
                fontsize=9,
                bbox=dict(
                    boxstyle="round",
                    facecolor="none",
                    edgecolor="grey",
                    alpha=0.5,
                ),
            )
            # config
            ax[i].set_ylabel(cov)
            ax[i].xaxis.set_tick_params(labelbottom=True)
            ax[i].set_yticks([])
            if bins is not None:
                ax[i].set_yticks([0, 1])
        ax[0].set_title(
            f"models = {len(learner_info)}/{2 ** len(summary)}", loc="left"
        )

        return fig

    # validations ==============================================================
    def _as_model_type(self, model_type: str) -> str:
        if model_type not in model_type_dict:
            raise InvalidConfigurationError(
                f"{model_type=:} not known, "
                f"please select from {list(model_type_dict.keys())}"
            )
        return model_type

    def _as_cov(
        self, cov_fixed: list[str], cov_exploring: list[str]
    ) -> tuple[list[str], list[str]]:
        len_set = len(set(cov_fixed) | set(cov_exploring))
        len_sum = len(cov_fixed) + len(cov_exploring)
        if len_set != len_sum:
            raise InvalidConfigurationError(
                "Covariates in cov_fixed and cov_exploring cannot repeat"
            )
        return list(cov_fixed), list(cov_exploring)

    def _as_main_param(self, main_param: str | None) -> str:
        params = self.params
        if main_param is not None:
            if main_param not in params:
                raise InvalidConfigurationError(
                    f"{main_param=:} not know, " f"please select from {params}"
                )
        else:
            if len(params) > 1:
                raise InvalidConfigurationError(
                    "There are more than one model parameters, "
                    f"please select main_param from {params}"
                )
            main_param = params[0]
        return main_param

    def _as_param_specs(
        self, param_specs: dict[str, dict] | None
    ) -> dict[str, dict]:
        param_specs = param_specs or {}
        for param in self.params:
            if param != self.main_param:
                variables = param_specs.get(param, {}).get("variables", [])
                if len(variables) == 0:
                    example_param_specs = {param: {"variables": ["intercept"]}}
                    raise InvalidConfigurationError(
                        f"Please provide variables for {param}, "
                        f"for example, param_specs={example_param_specs}"
                    )
        param_specs.update({self.main_param: {"variables": self.cov_fixed}})
        return param_specs

    # construct learner ========================================================
    def _get_param_specs(self, learner_id: LearnerID) -> dict[str, dict]:
        param_specs = deepcopy(self.param_specs)
        param_specs[self.main_param]["variables"].extend(
            [self.cov_exploring[i] for i in learner_id]
        )
        return param_specs

    def _can_share_variables(self) -> bool:
        """Whether learners may share cached Variable objects. Only the fast
        gaussian lightweight path is eligible: there a Learner only ever reads
        ``Variable.name``, so identical Variables can be constructed once per
        Rover instead of 9+ times per learner. Mirrors the eligibility checks
        of ``Learner._can_use_fast_linear`` plus the lightweight-model
        condition (``get_score is not None``)."""
        cached = getattr(self, "_share_variables_ok", None)
        if cached is None:
            main_spec = self.param_specs.get(self.main_param, {})
            inv_link = main_spec.get("inv_link")
            cached = (
                getattr(self, "get_score", None) is not None
                and self.main_param == "mu"
                and tuple(self.model_class.param_names) == ("mu",)
                and (inv_link is None or inv_link == "identity")
            )
            self._share_variables_ok = cached
        return cached

    def _get_variable(self, name: str) -> Variable:
        cache = getattr(self, "_variable_cache", None)
        if cache is None:
            cache = self._variable_cache = {}
        var = cache.get(name)
        if var is None:
            var = Variable(name)
            cache[name] = var
        return var

    def _get_param_specs_shared(self, learner_id: LearnerID) -> dict[str, dict]:
        """Same result as ``_get_param_specs`` (fresh spec dicts and variable
        lists per learner) but without the per-learner deepcopy, and with
        Variable objects drawn from the per-Rover cache."""
        param_specs = {}
        for param, spec in self.param_specs.items():
            new_spec = dict(spec)
            variables = [self._get_variable(name) for name in spec["variables"]]
            if param == self.main_param:
                variables.extend(
                    self._get_variable(self.cov_exploring[i])
                    for i in learner_id
                )
            new_spec["variables"] = variables
            param_specs[param] = new_spec
        return param_specs

    def _get_learner(
        self, learner_id: LearnerID, use_cache: bool = True
    ) -> Learner:
        if learner_id in self.learners and use_cache:
            return self.learners[learner_id]

        if _batched_solves_enabled() and self._can_share_variables():
            param_specs = self._get_param_specs_shared(learner_id)
        else:
            param_specs = self._get_param_specs(learner_id)
        return Learner(
            self.model_class,
            self.obs,
            self.main_param,
            param_specs,
            weights=self.weights,
            get_score=self.get_score,
        )

    # explore ==================================================================
    def _explore(
        self,
        data: DataFrame,
        strategies: list[str],
        strategy_options: dict | None = None,
    ):
        """Explore the entire tree of learners"""
        strategy_options = strategy_options or {}
        strategy_options = {
            strategy: strategy_options.get(strategy, {})
            for strategy in strategies
        }
        holdout_data = None
        full_linear_cache = None
        if self.holdouts:
            holdout_data, full_linear_cache = self._prepare_holdout_data(
                data, self.holdouts
            )

        for strategy in strategies:
            options = strategy_options[strategy]
            strategy = get_strategy(strategy)(num_covs=len(self.cov_exploring))
            curr_ids = {strategy.base_learner_id}
            while curr_ids:
                # collect this layer's unfitted learners in the same iteration
                # order as the original per-learner loop, so that the
                # self.learners insertion order (and hence learner_info /
                # learner-table row order) is unchanged
                pending: list[tuple[LearnerID, Learner]] = []
                for learner_id in curr_ids:
                    learner = self._get_learner(learner_id)
                    if learner.status == ModelStatus.NOT_FITTED:
                        pending.append((learner_id, learner))
                if pending:
                    if not self._fit_layer_batched(
                        pending, data, holdout_data, full_linear_cache
                    ):
                        for _, learner in pending:
                            learner.fit(
                                data,
                                self.holdouts,
                                holdout_data=holdout_data,
                                full_linear_cache=full_linear_cache,
                            )
                    for learner_id, learner in pending:
                        self.learners[learner_id] = learner

                next_ids = strategy.get_next_layer(
                    curr_layer=curr_ids,
                    learners=self.learners,
                    **options,
                )
                curr_ids = next_ids

    # batched layer solves =====================================================
    def _fit_layer_batched(
        self,
        pending: list[tuple[LearnerID, Learner]],
        data: DataFrame,
        holdout_data: dict | None,
        full_linear_cache: dict[str, Any] | None,
    ) -> bool:
        """Fit all learners of one exploration layer with stacked linear
        solves (one ``np.linalg.solve`` on a (C, s, s) Gram stack per
        holdout/full fit) instead of C independent solves.

        Only the fast gaussian lightweight path with gram-based validation
        scoring is eligible; returns False (caller falls back to the
        per-learner path) otherwise. The solves are the same LAPACK ``gesv``
        on the same matrices, so coefficients are bit-identical to the
        per-learner path; the vectorized SSE reductions (einsum/matmul) may
        reorder floating-point accumulation of the *scores* at the ~1e-15
        level. Set ``MODROVER_BATCHED_SOLVES=0`` to disable."""
        if not _batched_solves_enabled():
            return False
        if not self.holdouts or holdout_data is None:
            return False
        if full_linear_cache is None:
            return False
        col_index = full_linear_cache.get("col_index")
        if col_index is None:
            return False
        for _, learner in pending:
            if not (
                learner._use_fast_linear
                and learner._use_lightweight_model
                and learner.get_score is get_rmse
            ):
                return False
        caches = []
        for holdout in self.holdouts:
            entry = holdout_data.get(holdout)
            if entry is None:
                return False
            linear_cache = entry[3]
            if linear_cache is None or "val_xtx" not in linear_cache:
                return False
            caches.append(linear_cache)

        # group by learner size (layers are uniform for forward/backward, but
        # e.g. the full strategy and mixed strategies need the generality)
        groups: dict[int, list[Learner]] = {}
        for _, learner in pending:
            groups.setdefault(len(learner._main_cov_names), []).append(learner)
        for group in groups.values():
            for start in range(0, len(group), _BATCH_CHUNK):
                self._fit_group_batched(
                    group[start : start + _BATCH_CHUNK],
                    data,
                    caches,
                    full_linear_cache,
                    col_index,
                )
        return True

    @staticmethod
    def _solve_stacked(
        cache: dict[str, Any], idx: NDArray, group: list[Learner]
    ) -> tuple[NDArray, list[ModelStatus]]:
        """One batched solve for all children; on any singular child fall
        back to the exact per-learner solve path so that statuses and the
        lstsq near-singular rescue behave identically to the unbatched code."""
        xtwx = cache["xtwx"]
        xtwy = cache["xtwy"]
        gram = xtwx[idx[:, :, None], idx[:, None, :]]
        rhs = xtwy[idx]
        try:
            # rhs as an explicit stack of column vectors (nrhs=1 gesv, the
            # same LAPACK call the per-learner vector solve makes)
            coefs = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
            statuses = [ModelStatus.SUCCESS] * len(group)
        except np.linalg.LinAlgError:
            coefs = np.full(rhs.shape, np.nan)
            statuses = []
            for c, learner in enumerate(group):
                status, coef = learner._fit_fast_linear_coef(cache, idx[c])
                statuses.append(status)
                if status == ModelStatus.SUCCESS and coef is not None:
                    coefs[c] = coef
        return coefs, statuses

    def _fit_group_batched(
        self,
        group: list[Learner],
        data: DataFrame,
        caches: list[dict[str, Any]],
        full_linear_cache: dict[str, Any],
        col_index: dict[str, int],
    ) -> None:
        """Fit one same-size chunk of a layer: per-holdout stacked CV solves
        with vectorized gram scoring, then one stacked full-data solve+inverse
        for coefficients and vcov. Per-learner statuses, cv bookkeeping and
        early-stopping semantics replicate ``Learner.fit`` exactly."""
        idx = np.array(
            [
                [col_index[name] for name in learner._main_cov_names]
                for learner in group
            ],
            dtype=int,
        )  # (C, s)
        n_children = len(group)
        active = np.ones(n_children, dtype=bool)

        for holdout, cache in zip(self.holdouts, caches):
            if not active.any():
                break
            coefs, statuses = self._solve_stacked(cache, idx, group)
            scores = None
            if any(st == ModelStatus.SUCCESS for st in statuses):
                # ||y_v - X_v c||^2 = y'y - 2 c'(X_v'y_v) + c'(X_v'X_v)c for
                # all children at once; identical algebra to the per-learner
                # gram scoring, vectorized over the stack
                val_xtx = cache["val_xtx"]
                val_xty = cache["val_xty"]
                vg = val_xtx[idx[:, :, None], idx[:, None, :]]
                vy = val_xty[idx]
                lin = np.einsum("ij,ij->i", coefs, vy)
                quad = np.einsum(
                    "ij,ij->i",
                    coefs,
                    np.matmul(vg, coefs[:, :, None])[:, :, 0],
                )
                sse = cache["val_yty"] - 2.0 * lin + quad
                mse = np.maximum(sse, 0.0) / cache["n_val"]
                scores = np.exp(-np.sqrt(mse))
            for c, learner in enumerate(group):
                if not active[c]:
                    continue
                status = statuses[c]
                learner._cv_status[holdout] = status
                if status == ModelStatus.SUCCESS:
                    learner._cv_scores[holdout] = float(scores[c])
                else:
                    learner.status = ModelStatus.CV_FAILED
                    active[c] = False

        for c, learner in enumerate(group):
            if active[c]:
                learner.score = np.mean(list(learner._cv_scores.values()))
            learner._cv_models.clear()

        active_idx = np.flatnonzero(active)
        if len(active_idx) == 0:
            return
        idx_a = idx[active_idx]
        xtwx = full_linear_cache["xtwx"]
        xtwy = full_linear_cache["xtwy"]
        gram = xtwx[idx_a[:, :, None], idx_a[:, None, :]]
        rhs = xtwy[idx_a]
        try:
            coefs = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            # a singular child poisons the whole stack: refit those learners
            # through the exact per-learner full-fit path
            for c in active_idx:
                learner = group[c]
                learner.status = learner._fit(
                    data,
                    linear_cache=full_linear_cache,
                    linear_idx=idx[c],
                )
            return
        lin = np.einsum("ij,ij->i", coefs, rhs)
        quad = np.einsum(
            "ij,ij->i", coefs, np.matmul(gram, coefs[:, :, None])[:, :, 0]
        )
        wsse = np.maximum(
            float(full_linear_cache["ywy"]) - 2.0 * lin + quad, 0.0
        )
        dof = max(int(full_linear_cache["n_obs"]) - idx_a.shape[1], 1)
        sigma2 = wsse / dof
        vcov = sigma2[:, None, None] * gram_inv
        for row, c in enumerate(active_idx):
            learner = group[c]
            learner.model.opt_coefs = coefs[row].copy()
            learner.model.opt_vcov = vcov[row].copy()
            learner.status = ModelStatus.SUCCESS

    def _prepare_holdout_data(
        self, data: DataFrame, holdouts: list[str]
    ) -> tuple[
        dict[str, tuple[DataFrame, DataFrame, NDArray, dict[str, Any] | None]],
        dict[str, Any] | None,
    ]:
        """Precompute train/validation splits for all holdouts once."""
        if "offset" not in data.columns or "trim_weights" not in data.columns:
            data = data.copy()
            if "offset" not in data.columns:
                data["offset"] = 0.0
            if "trim_weights" not in data.columns:
                data["trim_weights"] = 1.0

        cov_names = list(dict.fromkeys([*self.cov_fixed, *self.cov_exploring]))
        col_index = {name: i for i, name in enumerate(cov_names)}
        full_linear_cache = (
            self._build_linear_cache(data, cov_names, col_index)
            if cov_names
            else None
        )

        split_data = {}
        for holdout in holdouts:
            train_data = data[data[holdout] == 0]
            val_data = data[data[holdout] == 1]
            if len(train_data) == 0 or len(val_data) == 0:
                raise InvalidConfigurationError(
                    f"Holdout {holdout} must include both 0 and 1 rows."
                )
            val_obs = val_data[self.obs].to_numpy(dtype=float, copy=False)
            linear_cache = None
            if cov_names:
                linear_cache = self._build_linear_cache(
                    train_data, cov_names, col_index
                )
                x_val = val_data[cov_names].to_numpy(dtype=float, copy=False)
                linear_cache["x_val"] = x_val
                if _gram_scoring_enabled():
                    # Unweighted validation Gram matrices: the holdout score
                    # (globals.get_rmse) is exp(-sqrt(mean((obs - pred)**2)))
                    # with NO weights, so the validation SSE for any coef c is
                    #   ||y_v - X_v c||^2
                    #     = y_v'y_v - 2 c'(X_v'y_v) + c'(X_v'X_v)c,
                    # computable in O(s^2) per learner from these one-time
                    # per-holdout Grams instead of O(n_val * s) predictions.
                    # (The weighted full-minus-train cache subtraction is only
                    # equivalent when weights*trim_weights == 1 everywhere;
                    # direct unweighted validation Grams are exact always and
                    # avoid large-magnitude cancellation.)
                    y_val = np.asarray(val_obs, dtype=float).reshape(-1)
                    linear_cache["val_xtx"] = x_val.T.dot(x_val)
                    linear_cache["val_xty"] = x_val.T.dot(y_val)
                    linear_cache["val_yty"] = float(y_val.dot(y_val))
                    linear_cache["n_val"] = int(len(y_val))
            split_data[holdout] = (train_data, val_data, val_obs, linear_cache)
        return split_data, full_linear_cache

    def _build_linear_cache(
        self,
        data: DataFrame,
        cov_names: list[str],
        col_index: dict[str, int],
    ) -> dict[str, Any]:
        x = data[cov_names].to_numpy(dtype=float, copy=False)
        y = data[self.obs].to_numpy(dtype=float, copy=False).reshape(-1)
        w = data[self.weights].to_numpy(dtype=float, copy=False).reshape(-1)
        if "trim_weights" in data.columns:
            w = w * data["trim_weights"].to_numpy(dtype=float, copy=False).reshape(
                -1
            )
        sqrt_w = np.sqrt(w)
        xw = x * sqrt_w[:, None]
        yw = y * sqrt_w

        return {
            "col_index": col_index,
            "xtwx": xw.T.dot(xw),
            "xtwy": xw.T.dot(yw),
            "ywy": float(yw.dot(yw)),
            "n_obs": int(len(y)),
        }

    # construct super learner ==================================================
    def _get_super_learner(
        self,
        top_pct_score: float,
        top_pct_learner: float,
        coef_bounds: dict[str, tuple[float, float]] | None,
    ) -> Learner:
        """Call at the end of fit, so model is configured at the end of fit."""
        df = self._get_learner_info(top_pct_score, top_pct_learner, coef_bounds)
        df = df[df["weight"] > 0.0]
        learner_ids, weights = df["learner_id"], df["weight"]
        coefs = df[list(self.variables)].to_numpy()
        super_coef = coefs.T.dot(weights)
        super_vcov = self._get_super_vcov(learner_ids, weights, super_coef)

        super_learner = self._get_learner(
            learner_id=self.super_learner_id, use_cache=False
        )
        super_learner.coef = super_coef
        super_learner.vcov = super_vcov
        self._super_learner = super_learner
        self._get_summary()
        return super_learner

    def _get_learner_info(
        self,
        top_pct_score: float = 0.1,
        top_pct_learner: float = 1.0,
        coef_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> DataFrame:
        learner_ids = list(self.learners.keys())
        n_learners = len(learner_ids)

        coef = np.full((n_learners, self.num_vars), np.nan, dtype=float)
        score = np.full(n_learners, np.nan, dtype=float)
        status = np.empty(n_learners, dtype=object)

        for i, learner_id in enumerate(learner_ids):
            learner = self.learners[learner_id]
            status[i] = learner.status
            if learner.status == ModelStatus.SUCCESS:
                coef_idx = self._get_coef_index(learner_id)
                coef_row = np.zeros(self.num_vars, dtype=float)
                coef_row[coef_idx] = learner.coef
                coef[i] = coef_row
                score[i] = learner.score

        df = DataFrame(coef, columns=list(self.variables))
        df.insert(0, "status", status)
        df.insert(0, "learner_id", learner_ids)
        df["score"] = score

        df["coef_valid"] = True
        if coef_bounds:
            coef_valid = []
            for cov, bounds in coef_bounds.items():
                if not any(map(cov.startswith, self.params)):
                    cov = "_".join([self.main_param, cov])
                coef_valid.append(
                    (df[cov] >= bounds[0]) & (df[cov] <= bounds[1])
                )
            df["coef_valid"] = np.vstack(coef_valid).all(axis=0)

        df["valid"] = (df["status"] == ModelStatus.SUCCESS) & df["coef_valid"]
        df["weight"] = 0.0
        valid_ids = df.loc[df["valid"], "learner_id"]
        if len(valid_ids) > 0:
            df.loc[df["valid"], "weight"] = self._get_super_weights(
                valid_ids, top_pct_score, top_pct_learner
            )

        score_max = df["score"].dropna().max()
        if np.isfinite(score_max) and score_max != 0.0:
            df["score_scaled"] = df["score"] / score_max
        else:
            df["score_scaled"] = np.nan
        self._learner_info = df
        return df

    def _get_super_coef(
        self, learner_ids: list[LearnerID], weights: NDArray
    ) -> NDArray:
        """Generates the weighted ensembled coefficients across all fitted
        learners.

        """
        super_coef = np.zeros(self.num_vars)
        for learner_id, weight in zip(learner_ids, weights):
            coef_index = self._get_coef_index(learner_id)
            super_coef[coef_index] += weight * self.learners[learner_id].coef
        return super_coef

    def _get_super_vcov(
        self,
        learner_ids: list[LearnerID],
        weights: NDArray,
        super_coef: NDArray,
    ) -> NDArray:
        super_vcov = np.zeros((self.num_vars, self.num_vars))
        for learner_id, weight in zip(learner_ids, weights):
            learner = self.learners[learner_id]
            coef_index = self._get_coef_index(learner_id)
            super_vcov[np.ix_(coef_index, coef_index)] += weight * (
                learner.vcov + np.outer(learner.coef, learner.coef)
            )
        super_vcov -= np.outer(super_coef, super_coef)
        return super_vcov

    def _get_coef_index(self, learner_id: LearnerID) -> list[int]:
        cached = self._coef_index_cache.get(learner_id)
        if cached is not None:
            return cached

        coef_index, pointer = [], 0
        for param in self.params:
            num_covs = len(self.param_specs[param]["variables"])
            coef_index.extend(list(range(pointer, pointer + num_covs)))
            pointer += num_covs
            if param == self.main_param:
                coef_index.extend([i + pointer for i in learner_id])
                pointer += len(self.cov_exploring)

        self._coef_index_cache[learner_id] = coef_index
        return coef_index

    def _get_super_weights(
        self,
        learner_ids: list[LearnerID],
        top_pct_score: float,
        top_pct_learner: float,
    ) -> NDArray:
        scores = np.array(
            [self.learners[learner_id].score for learner_id in learner_ids]
        )
        argsort = np.argsort(scores)[::-1]
        indices = scores >= scores[argsort[0]] * (1 - top_pct_score)
        num_learners = int(np.floor(len(scores) * top_pct_learner)) + 1
        indices[argsort[num_learners:]] = False

        scores[~indices] = 0.0
        weights = scores / scores.sum()
        return weights

    # diagnostics ==============================================================
    def _get_summary(self) -> DataFrame:
        # info
        variables = self.variables
        learner_info = self.learner_info[
            self.learner_info["status"] == ModelStatus.SUCCESS
        ]
        learner_scores = dict(
            zip(learner_info["learner_id"], learner_info["score"])
        )
        # ensemble info
        coef_index = [
            variables.index(f"{self.main_param}_{cov}")
            for cov in self.cov_exploring
        ]
        coef = self.super_learner.coef[coef_index]
        coef_sd = np.sqrt(np.diag(self.super_learner.vcov)[coef_index])
        # number of models the covariate is present
        pct_present = [
            (learner_info[f"{self.main_param}_{cov}"] != 0.0).sum()
            / len(learner_info)
            for cov in self.cov_exploring
        ]
        # score when only the selected covariate is present
        single_score = [
            learner_scores.get((i,), np.nan)
            for i in range(len(self.cov_exploring))
        ]
        # average score when selected covariate is present or not
        present_score = []
        not_present_score = []
        for cov in self.cov_exploring:
            present_index = learner_info[f"{self.main_param}_{cov}"] != 0.0
            ps, nps = 0.0, 0.0
            if any(present_index):
                ps = learner_info[present_index]["score"].mean()
            if not all(present_index):
                nps = learner_info[~present_index]["score"].mean()
            present_score.append(ps)
            not_present_score.append(nps)

        df = DataFrame(
            {
                "cov": self.cov_exploring,
                "coef": coef,
                "coef_sd": coef_sd,
                "pct_present": pct_present,
                "single_score": single_score,
                "present_score": present_score,
                "not_present_score": not_present_score,
            }
        )

        # derived quantities
        df["score_improvement"] = df["present_score"] / df["not_present_score"]
        df["ranking"] = (
            df["score_improvement"].rank(ascending=False).astype(int)
        )
        df["coef_lwr"] = coef - 1.96 * coef_sd
        df["coef_upr"] = coef + 1.96 * coef_sd
        df["significant"] = np.sign(df["coef_lwr"] * df["coef_upr"]) > 0
        self._summary = df
        return df
