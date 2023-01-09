from typing import Callable

from .globals import get_rmse
from .learner import Learner
from .learnerid import LearnerID


class Rover:

    def __init__(
        self,
        model_type: str,
        col_obs: str,
        col_weights: str,
        col_fixed: dict[str, list[str]],
        col_covs: dict[str, list[str]],
        col_offset: dict[str, str],
        model_eval_metric: Callable = get_rmse,
    ) -> None:
        self.model_type = model_type
        self.col_obs = col_obs
        self.col_weights = col_weights
        self.col_fixed = col_fixed
        self.col_covs = col_covs
        self.col_offset = col_offset
        self.model_eval_metric = model_eval_metric

        # TODO: validuate the inputs
        # ...

    def get_learner(self, learner_id: LearnerID) -> Learner:
        all_covariates = list(self.col_covs.values())[0]
        col_covs = {}
        for param_name, covs in self.col_fixed.items():
            col_covs[param_name] = covs.copy()
            if param_name in self.col_covs:
                col_covs[param_name].extend([
                    all_covariates[i - 1] for i in learner_id
                ])
        return Learner(
            learner_id,
            self.model_type,
            self.col_obs,
            self.col_covs,
            self.col_offset,
            self.col_weights,
            self.model_eval_metric,
        )
