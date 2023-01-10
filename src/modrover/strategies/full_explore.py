from itertools import combinations
from typing import Generator

from modrover.learnerid import LearnerID
from modrover.strategies.base import RoverStrategy


class FullExplore(RoverStrategy):

    def __init__(self, num_covariates: int):
        super().__init__(num_covariates)

    def generate_next_layer(self, *args, **kwargs) -> Generator:
        """Find every single possible learner ID combination, return in a single layer."""
        all_learner_ids = list(range(self.num_covariates + 1))
        for num_elements in range(1, self.num_covariates + 1):
            yield from combinations(all_learner_ids, num_elements)

    def get_upstream_learner_ids(self, learner_id: LearnerID):
        """This method is irrelevant for full explore.

        There are no dependencies, we just fit every single combination of covariates."""
        raise NotImplementedError
