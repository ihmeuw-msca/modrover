from typing import Callable, Optional, Union

import pandas as pd

from .globals import get_rmse
from .learner import Learner
from .learnerid import LearnerID
from .strategies.base import RoverStrategy
from .strategies import get_strategy


class Rover:

    def __init__(
        self,
        model_type: str,
        y: str,
        cov_fixed: dict[str, list[str]],
        cov_explore: dict[str, list[str]],
        extra_param_specs: Optional[dict[str, dict]] = None,
        offset: str = "offset",
        weights: str = "weights",
        holdout_cols: Optional[list[str]] = None,
        model_eval_metric: Callable = get_rmse,
    ) -> None:
        # parse extra_param_specs
        if extra_param_specs is None:
            extra_param_specs = {}

        self.model_type = model_type
        self.y = y
        self.cov_fixed = cov_fixed
        self.cov_explore = cov_explore
        self.extra_param_specs = extra_param_specs
        self.offset = offset
        self.weights = weights
        self.holdout_cols = holdout_cols
        self.model_eval_metric = model_eval_metric

        self.learners: dict[LearnerID, Learner] = {}

        # TODO: validate the inputs
        # ...

    @property
    def performances(self):
        """Convenience property to select only the performance of each learner."""
        return {lid: learner.performance for lid, learner in self.learners.items()}

    def get_learner(self, learner_id: LearnerID) -> Learner:

        # See if we've already initialized one
        if learner_id in self.learners:
            return self.learners[learner_id]

        all_covariates = list(self.cov_explore.values())[0]
        param_specs = {}
        for param_name, covs in self.cov_fixed.items():
            variables = covs.copy()
            if param_name in self.cov_explore:
                variables.extend([
                    # Ignore the always-present 0 index. Results in duplicate column names.
                    all_covariates[i - 1] for i in learner_id.cov_ids[1:]
                ])
            param_specs[param_name] = {}
            param_specs[param_name]["variables"] = variables
            param_specs[param_name].update(
                self.extra_param_specs.get(param_name, {})
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

    def explore(self, dataset: pd.DataFrame, strategy: Union[str, RoverStrategy]):
        """Explore the entire tree of learners.

        Params:
        dataset: The dataset to fit models on
        strategy: a string or a roverstrategy object. Dictates how the next set of models
            is selected
        """
        # TODO: explore or fit?
        if isinstance(strategy, str):
            # If a string is provided, select a strategy for the user.
            strategy_class = get_strategy(strategy)
            strategy = strategy_class(
                num_covariates=len(list(self.cov_explore.values())[0]),
            )

        current_ids = {strategy.base_learnerid}

        while current_ids:
            current_learners = []
            for learner_id in current_ids:
                learner = self.get_learner(learner_id)

                if not learner.has_been_fit:
                    current_learners.append(learner)

            self._fit_layer(dataset, current_learners)
            next_ids = strategy.generate_next_layer(
                current_learner_ids=current_ids,
                prior_learners=self.learners
            )
            current_ids = set(next_ids)
        return

    def _fit_layer(self, dataset: pd.DataFrame, learners: list[Learner]) -> None:
        """Fit a layer of models. Store results on the learners dict."""
        for learner in learners:
            print(learner)
            learner.fit(dataset, self.holdout_cols)
            self.learners[learner.learner_id] = learner
