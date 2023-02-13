from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .exceptions import NotFittedError, InvalidConfigurationError
from .globals import get_rmse
from .learner import Learner, LearnerID
from .strategies import get_strategy
from .strategies.base import RoverStrategy
from .synthesizer import metrics_to_weights


class Rover:

    def __init__(
        self,
        model_type: str,
        y: str,
        col_fixed: dict[str, list[str]],
        col_explore: dict[str, list[str]],
        param_specs: Optional[dict[str, dict]] = None,
        offset: str = "offset",
        weights: str = "weights",
        holdout_cols: Optional[list[str]] = None,
        model_eval_metric: Callable = get_rmse,
    ) -> None:
        # parse param_specs
        if param_specs is None:
            param_specs = {}

        self.model_type = model_type
        self.y = y
        self.col_fixed = col_fixed
        self.col_explore = col_explore
        self.param_specs = param_specs
        self.offset = offset
        self.weights = weights
        self.holdout_cols = holdout_cols
        self.model_eval_metric = model_eval_metric

        self.learners: dict[LearnerID, Learner] = {}

        self.super_learner = None

        # TODO: validate the inputs
        # ...

    @property
    def performances(self):
        """Convenience property to select only the performance of each learner."""
        return {lid: learner.performance for lid, learner in self.learners.items()}

    @property
    def num_covariates(self):
        # TODO: Validate this method, check for overlaps and such
        num_covariates = 0
        for key, val in self.col_fixed.items():
            num_covariates += len(val)

        for key, val in self.col_explore.items():
            num_covariates += len(val)

        return num_covariates

    def check_is_fitted(self):
        if not self.super_learner:
            raise NotFittedError("Rover has not been ensembled yet")

    def get_learner(self, learner_id: LearnerID, use_cache: bool = True) -> Learner:

        # See if we've already initialized one
        if learner_id in self.learners and use_cache:
            return self.learners[learner_id]

        all_covariates = list(self.col_explore.values())[0]
        param_specs = {}
        for param_name, covs in self.col_fixed.items():
            variables = covs.copy()
            if param_name in self.col_explore:
                variables.extend([
                    # Ignore the always-present 0 index. Results in duplicate column names.
                    all_covariates[i - 1] for i in learner_id[1:]
                ])
            param_specs[param_name] = {}
            param_specs[param_name]["variables"] = variables
            param_specs[param_name].update(
                self.param_specs.get(param_name, {})
            )
        return Learner(
            learner_id,
            self.model_type,
            self.y,
            param_specs,
            offset=self.offset,
            weights=self.weights,
            model_eval_metric=self.model_eval_metric,
        )

    def fit(
        self,
        dataset: pd.DataFrame,
        strategy: Union[str, RoverStrategy],
        max_num_models: int = 10,
        kernel_param: float = 10.,
        ratio_cutoff: float = .99
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
        self._explore(
            dataset=dataset,
            strategy=strategy
        )
        super_learner = self._create_super_learner(
            max_num_models=max_num_models,
            kernel_param=kernel_param,
            ratio_cutoff=ratio_cutoff
        )
        self.super_learner = super_learner

    def _explore(self, dataset: pd.DataFrame, strategy: Union[str, RoverStrategy]):
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
                num_covs=len(list(self.col_explore.values())[0]),
            )

        curr_ids = {strategy.base_learner_id}

        while curr_ids:
            curr_learners = []
            for learner_id in curr_ids:
                learner = self.get_learner(learner_id)

                if not learner.has_been_fit:
                    curr_learners.append(learner)

            self._fit_layer(dataset, curr_learners)
            next_ids = strategy.get_next_layer(
                curr_layer=curr_ids,
                learners=self.learners
            )
            curr_ids = set(next_ids)
        return

    def _fit_layer(self, dataset: pd.DataFrame, learners: list[Learner]) -> None:
        """Fit a layer of models. Store results on the learners dict."""
        for learner in learners:
            learner.fit(dataset, self.holdout_cols)
            self.learners[learner.learner_id] = learner

    def _generate_ensemble_coefficients(
        self,
        max_num_models: int,
        kernel_param: float,
        ratio_cutoff: float,
    ) -> np.ndarray:
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

        learner_ids, coefficients = self._generate_coefficients_matrix()

        # Create weights
        performances = np.array([self.performances[key] for key in learner_ids])
        weights = metrics_to_weights(
            metrics=performances,
            max_num_models=max_num_models,
            kernel_param=kernel_param,
            ratio_cutoff=ratio_cutoff
        )
        # Multiply by the coefficients
        means = coefficients.T.dot(weights)

        return means

    def _generate_coefficients_matrix(self) -> tuple[list[LearnerID], np.ndarray]:
        """Create the full matrix of learner ids, mapped to its relative coefficients.

        Will result in a m x n matrix, where m = the number of fitted learners in rover
        and n is the total number of covariates provided.

        Each cell is the i-th learner's coefficient for the j-th covariate, defaulting to 0
        if the coefficient is not represented in that learner.

        :return:
        """
        learner_ids: list[LearnerID] = []
        performances: np.ndarray = np.array([])
        for learner_id, learner in self.learners.items():

            if learner.performance and learner.opt_coefs is not None:
                learner_ids.append(learner_id)
                performances = np.append(performances, learner.performance)

        # Aggregate the coefficients
        x_dim, y_dim = len(learner_ids), self.num_covariates
        coefficients = np.zeros((x_dim, y_dim))

        for row, learner_id in enumerate(learner_ids):
            opt_coefs = self.learners[learner_id].opt_coefs
            coefficients[row][list(learner_id)] = opt_coefs

        return learner_ids, coefficients

    def _create_super_learner(
        self,
        max_num_models: int,
        kernel_param: float,
        ratio_cutoff: float
    ) -> Learner:
        """Call at the end of fit, so model is configured at the end of fit."""
        means = self._generate_ensemble_coefficients(
            max_num_models=max_num_models,
            kernel_param=kernel_param,
            ratio_cutoff=ratio_cutoff
        )
        master_learner_id = tuple(range(self.num_covariates))
        learner = self.get_learner(
            learner_id=master_learner_id,
            use_cache=False
        )
        learner.opt_coefs = means
        return learner

    def predict(self, dataset):
        self.check_is_fitted()
        return self.super_learner.predict(dataset)

    def summary(self):
        NotImplemented
