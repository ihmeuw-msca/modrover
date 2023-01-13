from typing import Callable, Optional

from .globals import get_rmse
from .learner import Learner
from .learnerid import LearnerID


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
        self.model_eval_metric = model_eval_metric

        # TODO: validate the inputs
        # ...

    def get_learner(self, learner_id: LearnerID) -> Learner:
        all_covariates = list(self.cov_explore.values())[0]
        param_specs = {}
        for param_name, covs in self.cov_fixed.items():
            variables = covs.copy()
            if param_name in self.cov_explore:
                variables.extend([
                    all_covariates[i - 1] for i in learner_id.cov_ids
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
