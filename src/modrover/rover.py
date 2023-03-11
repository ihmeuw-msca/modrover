from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from .exceptions import InvalidConfigurationError, NotFittedError
from .globals import get_rmse, model_type_dict
from .learner import Learner, LearnerID
from .strategies import get_strategy
from .strategies.base import RoverStrategy
from .synthesizer import metrics_to_weights


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
    model_eval_metric
        A callable used to evaluate cross-validated performance of sub-learners
        in rover

    """

    def __init__(
        self,
        model_type: str,
        obs: str,
        param: str,
        cov_fixed: list[str],
        cov_exploring: list[str],
        param_specs: Optional[dict[str, dict]] = None,
        offset: str = "offset",
        weights: str = "weights",
        holdouts: Optional[list[str]] = None,
        model_eval_metric: Callable = get_rmse,
    ) -> None:
        # parse param_specs
        if param_specs is None:
            param_specs = {}

        self.model_type = model_type
        self.obs = obs
        self.param = param
        self.cov_fixed = cov_fixed
        self.cov_exploring = cov_exploring

        self.param_specs = param_specs
        self.offset = offset
        self.weights = weights
        self.holdouts = holdouts
        self.model_eval_metric = model_eval_metric

        self.learners: dict[LearnerID, Learner] = {}

        self.super_learner = None

        # TODO: validate the inputs
        # ...

    @property
    def num_covs(self) -> int:
        num_cov = len(self.cov_exploring) + sum(
            len(self.param_specs[param]["variables"])
            for param in self.model_class.param_names
        )
        return num_cov

    @property
    def model_class(self) -> type:
        if self.model_type not in model_type_dict:
            raise InvalidConfigurationError(
                f"Model type {self.model_type} not known, "
                f"please select from {list(model_type_dict.keys())}"
            )
        return model_type_dict[self.model_type]

    @property
    def param_specs(self) -> dict[str, dict]:
        return self._param_specs

    @param_specs.setter
    def param_specs(self, param_specs: Optional[dict[str, dict]]):
        if param_specs is None:
            param_specs = {}
        param_specs.update({self.param: {"variables": self.cov_fixed}})
        self._param_specs = param_specs

    def check_is_fitted(self):
        if not self.super_learner:
            raise NotFittedError("Rover has not been ensembled yet")

    def _get_param_specs(self) -> dict[str, dict]:
        param_specs = {}
        if self._param_specs is not None:
            param_specs = self._param_specs.copy()
        param_specs.update({self.param: {"variables": self.cov_fixed.copy()}})
        return param_specs

    def _get_learner(self, learner_id: LearnerID, use_cache: bool = True) -> Learner:
        if learner_id in self.learners and use_cache:
            return self.learners[learner_id]

        param_specs = deepcopy(self.param_specs)
        param_specs[self.param]["variables"].extend(
            [self.cov_exploring[i] for i in learner_id]
        )
        return Learner(
            self.model_class,
            self.obs,
            param_specs,
            offset=self.offset,
            weights=self.weights,
            model_eval_metric=self.model_eval_metric,
        )

    def fit(
        self,
        dataset: DataFrame,
        strategy: Union[str, RoverStrategy],
        max_num_models: int = 10,
        kernel_param: float = 10.0,
        ratio_cutoff: float = 0.99,
    ) -> None:
        """
        Fits the ensembled super learner.

        Explores over all covariate slices as defined by the input strategy, and fits the
        sublearners.

        The superlearner coefficients are determined by the ensemble method parameters, and the
        super learner itself will be created - to be used in prediction and summarization.

        :param dataset: the dataset to fit individual learners on
        :param strategy: the selection strategy to determine the model tree
        :param max_num_models: the maximum number of models to consider for ensembling
        :param kernel_param: the kernel parameter used to determine bias in ensemble weights
        :param ratio_cutoff: the cross-validated performance score necessary for a learner to
            be considered in ensembling
        :return: None
        """
        self._explore(dataset=dataset, strategy=strategy)
        super_learner = self._get_super_learner(
            max_num_models=max_num_models,
            kernel_param=kernel_param,
            ratio_cutoff=ratio_cutoff,
        )
        self.super_learner = super_learner

    def _explore(self, dataset: DataFrame, strategy: Union[str, RoverStrategy]):
        """Explore the entire tree of learners.

        Params:
        dataset: The dataset to fit models on
        strategy: a string or a roverstrategy object. Dictates how the next set of models
            is selected
        """
        if isinstance(strategy, str):
            # If a string is provided, select a strategy for the user.
            strategy_class = get_strategy(strategy)
            strategy = strategy_class(
                num_covs=len(self.cov_exploring),
            )

        curr_ids = {strategy.base_learner_id}

        while curr_ids:
            for learner_id in curr_ids:
                learner = self._get_learner(learner_id)
                if not learner.has_been_fit:
                    learner.fit(dataset, self.holdouts)
                    self.learners[learner_id] = learner

            next_ids = strategy.get_next_layer(
                curr_layer=curr_ids, learners=self.learners
            )
            curr_ids = next_ids
        return

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
            performances
        :param ratio_cutoff: The performance floor which learners must exceed to be considered
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
        performances = np.array([self.learners[key].performance for key in learner_ids])
        weights = metrics_to_weights(
            metrics=performances,
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
            if learner.has_been_fit
        ]

        # collect the coefficients from all models
        coef_mat = np.zeros((len(learner_ids), self.num_covs))
        for row, learner_id in enumerate(learner_ids):
            coef_index = self._get_coef_index(learner_id)
            coef_mat[row, coef_index] = self.learners[learner_id].coef

        return learner_ids, coef_mat

    def _get_coef_index(self, learner_id: LearnerID) -> tuple[int, ...]:
        coef_index, pointer = [], 0
        for param in self.model_class.param_names:
            num_covs = len(self.param_specs[param]["variables"])
            coef_index.extend(list(range(pointer, pointer + num_covs)))
            pointer += num_covs
            if param == self.param:
                coef_index.extend([i + pointer for i in learner_id])
                pointer += len(self.cov_exploring)
        return coef_index

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

    def predict(self, dataset):
        self.check_is_fitted()
        return self.super_learner.predict(dataset)

    def summary(self):
        NotImplemented
