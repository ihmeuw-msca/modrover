from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from .exceptions import InvalidConfigurationError, NotFittedError
from .globals import get_rmse, model_type_dict
from .learner import Learner, LearnerID, ModelStatus
from .strategies import get_strategy


class Rover:
    """Rover class explores model space and creates final super learner for
    prediction and inference.

    Parameters
    ----------
    model_type
        Type of the model. For example `"gaussian"` or `"poisson"`
    obs
        The name of the column representing observations
    param
        The parameter we are exploring over
    cov_fixed
        A list representing the covariates are present in every learner
    cov_exploring
        A list representing the covariates rover will explore over
    param_specs
        Parameter settings including, link function, priors, etc
    offset
        Offset of the model
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
        main_param: Optional[str] = None,
        param_specs: Optional[dict[str, dict]] = None,
        offset: str = "offset",
        weights: str = "weights",
        holdouts: Optional[list[str]] = None,
        get_score: Callable = get_rmse,
    ) -> None:
        self.model_type = self._as_model_type(model_type)
        self.obs = obs
        self.cov_fixed, self.cov_exploring = self._as_cov(cov_fixed, cov_exploring)
        self.main_param = self._as_main_param(main_param)
        self.param_specs = self._as_param_specs(param_specs)
        self.offset = offset
        self.weights = weights
        self.holdouts = holdouts
        self.get_score = get_score

        self.learners: dict[LearnerID, Learner] = {}

    @property
    def model_class(self) -> type:
        return model_type_dict[self.model_type]

    @property
    def params(self) -> tuple[str, ...]:
        return self.model_class.param_names

    @property
    def super_learner(self) -> Learner:
        if not hasattr(self, "_super_learner"):
            raise NotFittedError("Rover has not been ensembled yet")
        return self._super_learner

    def fit(
        self,
        data: DataFrame,
        strategy: str,
        max_num_models: int = 10,
        kernel_param: float = 10.0,
        ratio_cutoff: float = 0.99,
    ) -> None:
        """Fits the ensembled super learner.

        Explores over all covariate slices as defined by the input strategy, and
        fits the sublearners.

        The superlearner coefficients are determined by the ensemble method
        parameters, and the super learner itself will be created - to be used in
        prediction and summarization.

        Parameters
        ----------
        data
            Training data to fit individual learners on
        strategy
            The selection strategy to determine the model tree
        max_num_models
            The maximum number of models to consider for ensembling
        kernel_param
            The kernel parameter used to determine bias in ensemble weights
        ratio_cutoff
            The cross-validated score score necessary for a learner to be
            considered in ensembling

        """
        self._explore(data=data, strategy=strategy)
        super_learner = self._get_super_learner(
            max_num_models=max_num_models,
            kernel_param=kernel_param,
            ratio_cutoff=ratio_cutoff,
        )
        self._super_learner = super_learner

    def predict(self, data: DataFrame) -> NDArray:
        """Predict with ensembled super learner.

        Parameters
        ----------
        data
            Testing data to predict

        Returns
        -------
        NDArray
            Super learner predictions

        """
        return self.super_learner.predict(data)

    def summary(self):
        NotImplemented

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

    def _as_main_param(self, main_param: Optional[str]) -> str:
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
        self, param_specs: Optional[dict[str, dict]]
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

    def _get_learner(self, learner_id: LearnerID, use_cache: bool = True) -> Learner:
        if learner_id in self.learners and use_cache:
            return self.learners[learner_id]

        param_specs = self._get_param_specs(learner_id)
        return Learner(
            self.model_class,
            self.obs,
            param_specs,
            offset=self.offset,
            weights=self.weights,
            get_score=self.get_score,
        )

    # explore ==================================================================
    def _explore(self, data: DataFrame, strategy: str):
        """Explore the entire tree of learners.

        Params:
        dataset: The dataset to fit models on
        strategy: a string or a roverstrategy object. Dictates how the next set of models
            is selected
        """
        strategy = get_strategy(strategy)(num_covs=len(self.cov_exploring))

        curr_ids = {strategy.base_learner_id}

        while curr_ids:
            for learner_id in curr_ids:
                learner = self._get_learner(learner_id)
                if learner.status == ModelStatus.NOT_FITTED:
                    learner.fit(data, self.holdouts)
                    self.learners[learner_id] = learner

            next_ids = strategy.get_next_layer(
                curr_layer=curr_ids, learners=self.learners
            )
            curr_ids = next_ids

    # construct super learner ===================================================
    def _get_super_learner(
        self, max_num_models: int, kernel_param: float, ratio_cutoff: float
    ) -> Learner:
        """Call at the end of fit, so model is configured at the end of fit."""
        means = self._get_super_coef(
            max_num_models=max_num_models,
            kernel_param=kernel_param,
            ratio_cutoff=ratio_cutoff,
        )
        master_learner_id = tuple(range(len(self.cov_exploring)))
        learner = self._get_learner(learner_id=master_learner_id, use_cache=False)
        learner.coef = means
        return learner

    def _get_super_coef(
        self,
        max_num_models: int,
        kernel_param: float,
        ratio_cutoff: float,
    ) -> NDArray:
        """
        Generates the weighted ensembled coefficients across all fitted learners.

        :param max_num_models: The maximum number of learners to consider for our weights
        :param kernel_param: The kernel parameter, amount with which to bias towards strongest
            scores
        :param ratio_cutoff: The score floor which learners must exceed to be considered
            in the ensemble weights
        :return: A vector of weighted coefficients aggregated from sufficiently-performing
            sublearners.
        """

        # Validate the parameters
        if ratio_cutoff > 1 or max_num_models < 1:
            raise InvalidConfigurationError(
                "The ratio cutoff parameter must be < 1, and max_num_models >= 1, "
                "otherwise no models will be used for ensembling."
            )

        learner_ids, coef_mat = self._get_coef_mat()

        # Create weights
        scores = np.array([self.learners[key].score for key in learner_ids])
        weights = scores_to_weights(
            metrics=scores,
            max_num_models=max_num_models,
            kernel_param=kernel_param,
            ratio_cutoff=ratio_cutoff,
        )
        # Multiply by the coefficients
        super_coef = coef_mat.T.dot(weights)

        return super_coef

    def _get_coef_mat(self) -> tuple[list[LearnerID], NDArray]:
        """Create the full matrix of learner ids, mapped to its relative coefficients.

        Will result in a m x n matrix, where m = the number of fitted learners in rover
        and n is the total number of covariates provided.

        Each cell is the i-th learner's coefficient for the j-th covariate, defaulting to 0
        if the coefficient is not represented in that learner.

        :return:
        """
        learner_ids = [
            learner_id
            for learner_id, learner in self.learners.items()
            if learner.status == ModelStatus.SUCCESS
        ]

        # collect the coefficients from all models
        num_vars = len(self.cov_exploring) + sum(
            len(self.param_specs[param]["variables"]) for param in self.params
        )
        coef_mat = np.zeros((len(learner_ids), num_vars))
        for row, learner_id in enumerate(learner_ids):
            coef_index = self._get_coef_index(learner_id)
            coef_mat[row, coef_index] = self.learners[learner_id].coef

        return learner_ids, coef_mat

    def _get_coef_index(self, learner_id: LearnerID) -> list[int]:
        coef_index, pointer = [], 0
        for param in self.params:
            num_covs = len(self.param_specs[param]["variables"])
            coef_index.extend(list(range(pointer, pointer + num_covs)))
            pointer += num_covs
            if param == self.main_param:
                coef_index.extend([i + pointer for i in learner_id])
                pointer += len(self.cov_exploring)
        return coef_index


def scores_to_weights(
    metrics: NDArray,
    max_num_models: int,
    kernel_param: float,
    ratio_cutoff: float,
) -> NDArray:
    # Drop performances that aren't in the top n or don't meet a threshold
    # from being assigned weights
    sort_indices = np.argsort(metrics)[::-1]
    indices = metrics >= metrics.max() * ratio_cutoff
    indices[sort_indices[max_num_models:]] = False

    # Initialize weights vector
    weights = np.zeros(metrics.size)
    metrics = metrics[indices]

    metrics = metrics / metrics.max()
    # Apply exponential transform to selected performances
    metrics = np.exp(kernel_param * metrics)
    # Final weights should all sum to 1
    weights[indices] = metrics / metrics.sum()
    return weights
