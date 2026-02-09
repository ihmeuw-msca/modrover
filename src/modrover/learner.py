from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable
from scipy.stats import norm

LearnerID = tuple[int, ...]


class ModelStatus(Enum):
    SUCCESS = 0
    SINGULAR = 1
    CV_FAILED = 2
    SOLVER_FAILED = 3
    NOT_FITTED = -1


class Learner:
    """Individual learner class for one specific covariate configuration.

    Parameters
    ----------
    model_class
        Regmod model constructor
    obs
        Name corresponding to the observation column in the data frame
    main_param
        The main parameter we are exploring. This is aligned with the :class:`modrover.rover.Rover` class.
    param_specs
        Parameter settings for the regmod model
    weights
        Name corresponding to the weights column in the data frame
    get_score
        Function that evaluate the score of of the model

    """

    def __init__(
        self,
        model_class: type,
        obs: str,
        main_param: str,
        param_specs: dict[str, dict],
        weights: str = "weights",
        get_score: Callable | None = None,
    ) -> None:
        self.model_class = model_class
        self.obs = obs
        self.main_param = main_param
        self.weights = weights
        self.get_score = get_score

        # convert str to Variable
        for param_spec in param_specs.values():
            param_spec["variables"] = list(
                map(Variable, param_spec["variables"])
            )
        self.param_specs = param_specs
        self._main_cov_names = [
            var.name for var in self.param_specs[self.main_param]["variables"]
        ]
        self._use_fast_linear = self._can_use_fast_linear()

        # initialize null model
        self.model = self._get_model()
        self.score = np.nan
        self.status = ModelStatus.NOT_FITTED

        # initialize cross validation model
        self._cv_models = defaultdict(self._get_model)
        self._cv_scores = defaultdict(lambda: None)
        self._cv_status = defaultdict(lambda: ModelStatus.NOT_FITTED)

    @property
    def coef(self) -> NDArray | None:
        """Coefficients in the regmod model."""
        return self.model.opt_coefs

    @coef.setter
    def coef(self, coef: NDArray):
        if len(coef) != self.model.size:
            raise ValueError("Provided coef size don't match")
        self.model.opt_coefs = coef

    @property
    def vcov(self) -> NDArray | None:
        """Variance-covarianace matrix for the coefficients in the regmod model."""
        return self.model.opt_vcov

    @vcov.setter
    def vcov(self, vcov: NDArray):
        if vcov.shape != (self.model.size, self.model.size):
            raise ValueError("Provided vcov shape don't match")
        self.model.opt_vcov = vcov

    def fit(
        self,
        data: DataFrame,
        holdouts: list[str] | None = None,
        holdout_data: Mapping[
            str, tuple[DataFrame, DataFrame, NDArray, dict[str, Any] | None]
        ]
        | None = None,
        full_linear_cache: dict[str, Any] | None = None,
        **optimizer_options,
    ) -> None:
        """
        Fit a set of models on a series of holdout datasets.

        This method will fit a model over k folds of the dataset, where k is the
        length of the provided holdouts list. It is up to the user to decide the
        train-test splits for each holdout column.

        On each fold of the dataset, the trained model will predict out on the
        validation set and obtain a evaluate. The averaged evaluate across all
        folds becomes the model's overall score.

        Finally, a model is trained with all data in order to generate the final
        coefficients.

        Parameters
        ----------
        data
            A dataframe containing the training data
        holdouts
            Which column names to iterate over for cross validation. If it is
            `None`, insample performance score will be used to evaluate the
            model.
        holdout_data
            Optional precomputed mapping from holdout name to a tuple of
            (train_df, validation_df, validation_obs, linear_cache). This avoids
            recomputing split/groupby work and repeated observation extraction
            for every learner while preserving the same folds.
        full_linear_cache
            Optional precomputed linear algebra cache on the full dataset.
        **optimizer_options
            Extra options for the optimizer.

        """
        if self.status != ModelStatus.NOT_FITTED:
            return

        linear_idx = None
        if self._use_fast_linear and full_linear_cache is not None:
            col_index = full_linear_cache.get("col_index")
            if col_index is not None:
                linear_idx = np.array(
                    [col_index[name] for name in self._main_cov_names],
                    dtype=int,
                )

        if holdouts:
            # If holdout cols are provided, loop through to calculate OOS score
            for holdout in holdouts:
                linear_cache = None
                if holdout_data is not None and holdout in holdout_data:
                    train_data, val_data, val_obs, linear_cache = holdout_data[holdout]
                else:
                    data_group = data.groupby(holdout)
                    train_data = data_group.get_group(0)
                    val_data = data_group.get_group(1)
                    val_obs = val_data[self.obs].to_numpy()

                holdout_linear_idx = linear_idx
                if (
                    self._use_fast_linear
                    and holdout_linear_idx is None
                    and linear_cache is not None
                ):
                    col_index = linear_cache.get("col_index")
                    if col_index is not None:
                        holdout_linear_idx = np.array(
                            [col_index[name] for name in self._main_cov_names],
                            dtype=int,
                        )

                self._cv_status[holdout] = self._fit(
                    train_data,
                    self._cv_models[holdout],
                    linear_cache=linear_cache,
                    linear_idx=holdout_linear_idx,
                    **optimizer_options,
                )
                if self._cv_status[holdout] == ModelStatus.SUCCESS:
                    if self.get_score is None:
                        self._cv_scores[holdout] = self.evaluate(
                            val_data, self._cv_models[holdout]
                        )
                    else:
                        if (
                            self._use_fast_linear
                            and linear_cache is not None
                            and holdout_linear_idx is not None
                            and "x_val" in linear_cache
                        ):
                            x_val = linear_cache["x_val"][:, holdout_linear_idx]
                            coef = self._cv_models[holdout].opt_coefs
                            pred = x_val.dot(coef)
                        else:
                            pred = self.predict(
                                val_data, model=self._cv_models[holdout]
                            )
                        self._cv_scores[holdout] = self.get_score(
                            obs=val_obs, pred=pred
                        )
                else:
                    self.status = ModelStatus.CV_FAILED
                    break
            if self.status != ModelStatus.CV_FAILED:
                self.score = np.mean(list(self._cv_scores.values()))
            # clear all cv models for storage efficiency
            self._cv_models.clear()

        # Fit final model with all data included
        if self.status != ModelStatus.CV_FAILED:
            self.status = self._fit(
                data,
                linear_cache=full_linear_cache,
                linear_idx=linear_idx,
                **optimizer_options,
            )
            # If holdout cols not provided, use in-sample evaluate for the full data model
            if self.status == ModelStatus.SUCCESS and (not holdouts):
                self.score = self.evaluate(data)

    def predict(
        self,
        data: DataFrame,
        model: RegmodModel | None = None,
        return_ui: bool = False,
        alpha: float = 0.05,
    ) -> NDArray:
        """Generate prediction using regmod model. This function will return
        predictions for the :code:`main_param` with given data.

        Parameters
        ----------
        data
            A dataset to generate predictions from
        model
            A fitted RegmodModel. If it is ``None``, will use the overall model
            rather than the cross-validation model.
        return_ui
            If ``return_ui=True``, a matrix will be returned. The first row
            is the point prediction, second and thrid rows are the lower and
            upper bound of the prediction.
        alpha
            When ``return_ui=True``, function will return (1 - ``alpha``)
            uncertainty interval. By default, ``alpha=0.05``.

        """
        model = model or self.model
        if self._use_fast_linear and model.opt_coefs is not None:
            mat = data[self._main_cov_names].to_numpy(dtype=float)
            coef = model.opt_coefs
            pred = mat.dot(coef)
            if return_ui:
                if alpha < 0 or alpha > 0.5:
                    raise ValueError("`alpha` has to be between 0 and 0.5")
                vcov = model.opt_vcov
                if vcov is None:
                    raise ValueError("vcov is not available for uncertainty interval")
                lin_param_sd = np.sqrt((mat.dot(vcov) * mat).sum(axis=1))
                lin_param_lower = norm.ppf(
                    0.5 * alpha, loc=pred, scale=lin_param_sd
                )
                lin_param_upper = norm.ppf(
                    1 - 0.5 * alpha, loc=pred, scale=lin_param_sd
                )
                pred = np.vstack([pred, lin_param_lower, lin_param_upper])
            return pred

        model.data.attach_df(data)
        index = model.param_names.index(self.main_param)
        param = model.params[index]

        coef_index = model.indices[index]
        coef = model.opt_coefs[coef_index]

        offset = np.zeros(len(data))
        if param.offset is not None:
            offset = data[param.offset].to_numpy()

        mat = param.get_mat(model.data)
        lin_param = offset + mat.dot(coef)
        pred = param.inv_link.fun(lin_param)

        if return_ui:
            if alpha < 0 or alpha > 0.5:
                raise ValueError("`alpha` has to be between 0 and 0.5")
            vcov = model.opt_vcov[coef_index, coef_index]
            lin_param_sd = np.sqrt((mat.dot(vcov) * mat).sum(axis=1))
            lin_param_lower = norm.ppf(
                0.5 * alpha, loc=lin_param, scale=lin_param_sd
            )
            lin_param_upper = norm.ppf(
                1 - 0.5 * alpha, loc=lin_param, scale=lin_param_sd
            )
            pred = np.vstack(
                [
                    pred,
                    param.inv_link.fun(lin_param_lower),
                    param.inv_link.fun(lin_param_upper),
                ]
            )

        model.data.detach_df()
        return pred

    def evaluate(
        self, data: DataFrame, model: RegmodModel | None = None
    ) -> float:
        """Given a model and a test set, generate a performance score.

        Score is based on the provided evaluation metric, comparing the
        difference between observed and predicted values.

        Parameters
        ----------
        data
            The data set to generate predictions from
        model
            The fitted model to set predictions on. If ``None`` will use the
            overall model rather than the cross-validation model.

        """
        model = model or self.model
        if self.get_score is None:
            model.attach_df(data)
            score = np.exp(
                -model.objective(model.opt_coefs) / model.data.weights.sum()
            )
            model = _detach_df(model)
        else:
            score = self.get_score(
                obs=data[self.obs].to_numpy(),
                pred=self.predict(data, model=model),
            )
        return score

    def _get_model(self) -> RegmodModel:
        col_covs = []
        for param_spec in self.param_specs.values():
            col_covs.extend([var.name for var in param_spec["variables"]])
        col_covs = sorted(set(col_covs))

        # TODO: this shouldn't be necessary in regmod v1.0.0
        data = Data(
            col_obs=self.obs,
            col_covs=col_covs,
            col_weights=self.weights,
            subset_cols=True,
        )

        # Create regmod variables separately, by parameter
        # Initialize with fixed parameters
        model = self.model_class(
            data=data,
            param_specs=self.param_specs,
        )
        _enable_data_cache(model.data)
        return model

    def _fit(
        self,
        data: DataFrame,
        model: RegmodModel | None = None,
        linear_cache: dict[str, Any] | None = None,
        linear_idx: NDArray | None = None,
        **optimizer_options,
    ) -> ModelStatus:
        model = model or self.model
        if self._use_fast_linear:
            return self._fit_fast_linear(data, model, linear_cache, linear_idx)
        if "trim_weights" in data.columns:
            data["trim_weights"] = 1.0
        model.attach_df(data)
        try:
            model.fit(**optimizer_options)
            status = ModelStatus.SUCCESS
        except Exception as exc:
            msg = str(exc).lower()
            if isinstance(exc, np.linalg.LinAlgError) or "singular" in msg:
                status = ModelStatus.SINGULAR
            else:
                status = ModelStatus.SOLVER_FAILED
        model = _detach_df(model)
        return status

    def _fit_fast_linear(
        self,
        data: DataFrame,
        model: RegmodModel,
        linear_cache: dict[str, Any] | None = None,
        linear_idx: NDArray | None = None,
    ) -> ModelStatus:
        try:
            if linear_cache is not None and linear_idx is not None:
                idx = np.asarray(linear_idx, dtype=int)
                xtwx_all = linear_cache["xtwx"]
                xtwy_all = linear_cache["xtwy"]
                xtwx = xtwx_all[np.ix_(idx, idx)]
                xtwy = xtwy_all[idx]
                try:
                    coef = np.linalg.solve(xtwx, xtwy)
                except np.linalg.LinAlgError:
                    coef, _, rank, _ = np.linalg.lstsq(xtwx, xtwy, rcond=None)
                    if rank < xtwx.shape[1]:
                        return ModelStatus.SINGULAR
                n_obs = int(linear_cache["n_obs"])
                ywy = float(linear_cache["ywy"])
            else:
                x = data[self._main_cov_names].to_numpy(dtype=float)
                y = data[self.obs].to_numpy(dtype=float).reshape(-1)
                w = data[self.weights].to_numpy(dtype=float).reshape(-1)
                if "trim_weights" in data.columns:
                    w = w * data["trim_weights"].to_numpy(dtype=float).reshape(-1)

                sqrt_w = np.sqrt(w)
                xw = x * sqrt_w[:, None]
                yw = y * sqrt_w

                coef, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
                if rank < xw.shape[1]:
                    return ModelStatus.SINGULAR
                xtwx = xw.T.dot(xw)
                xtwy = xw.T.dot(yw)
                ywy = float(yw.dot(yw))
                n_obs = len(y)

            model.opt_coefs = coef
            # CV learners only need coefficients for scoring. We keep full vcov
            # computation for the final learner used in summary/inference.
            if model is self.model:
                wsse = max(
                    ywy - 2.0 * float(coef.dot(xtwy)) + float(coef.dot(xtwx.dot(coef))),
                    0.0,
                )
                dof = max(n_obs - len(coef), 1)
                sigma2 = wsse / dof

                xtwx_inv = np.linalg.inv(xtwx)
                vcov = sigma2 * xtwx_inv
                model.opt_vcov = vcov
            return ModelStatus.SUCCESS
        except Exception as exc:
            msg = str(exc).lower()
            if isinstance(exc, np.linalg.LinAlgError) or "singular" in msg:
                return ModelStatus.SINGULAR
            return ModelStatus.SOLVER_FAILED

    def _can_use_fast_linear(self) -> bool:
        # Exact for gaussian identity-link models with main parameter "mu".
        if self.main_param != "mu":
            return False
        if tuple(self.model_class.param_names) != ("mu",):
            return False
        main_spec = self.param_specs.get(self.main_param, {})
        inv_link = main_spec.get("inv_link")
        if inv_link is not None and inv_link != "identity":
            return False
        return True


def _enable_data_cache(data: Data) -> None:
    """Cache repeated Data.get_cols lookups for a single attached dataframe."""
    if getattr(data, "_modrover_cache_enabled", False):
        return

    col_cache: dict[tuple[str, tuple[str, ...] | str], NDArray] = {}
    orig_detach = data.detach_df
    orig_get_cols = data.get_cols

    def _cache_key(cols: str | list[str]) -> tuple[str, tuple[str, ...] | str] | None:
        if isinstance(cols, str):
            if cols == "trim_weights":
                return None
            return ("str", cols)
        cols_tuple = tuple(cols)
        if "trim_weights" in cols_tuple:
            return None
        return ("list", cols_tuple)

    def attach_df_cached(df: DataFrame):
        col_cache.clear()
        data.df = df
        if "intercept" not in data.df.columns:
            data.df["intercept"] = 1.0
        if data.col_weights not in data.df.columns:
            data.df[data.col_weights] = 1.0
        if data.col_offset not in data.df.columns:
            data.df[data.col_offset] = 0.0
        if data.col_obs is not None:
            cols = data.col_obs if isinstance(data.col_obs, list) else [data.col_obs]
            for col in cols:
                if col not in data.df.columns:
                    data.df[col] = np.nan
        if "trim_weights" not in data.df.columns:
            data.df["trim_weights"] = 1.0
        data.check_cols()

    def detach_df_cached():
        col_cache.clear()
        return orig_detach()

    def get_cols_cached(cols: str | list[str]) -> NDArray:
        key = _cache_key(cols)
        if key is None:
            return orig_get_cols(cols)
        cached = col_cache.get(key)
        if cached is None:
            cached = orig_get_cols(cols)
            col_cache[key] = cached
        return cached

    data.attach_df = attach_df_cached  # type: ignore[method-assign]
    data.detach_df = detach_df_cached  # type: ignore[method-assign]
    data.get_cols = get_cols_cached  # type: ignore[method-assign]
    data._modrover_cache_enabled = True  # type: ignore[attr-defined]


def _detach_df(model: RegmodModel) -> RegmodModel:
    """Detach data and all the arrays from the regmod model."""
    model.data.detach_df()
    del model.mat
    del model.uvec
    del model.gvec
    del model.linear_uvec
    del model.linear_gvec
    del model.linear_umat
    del model.linear_gmat

    return model
