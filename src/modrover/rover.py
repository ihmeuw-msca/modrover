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
        col_fixed: dict[str, list],
        col_covs: dict[str, list],
        col_offset: dict[str, list],
        model_eval_metric: Callable = get_rmse,
    ):
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
        ...