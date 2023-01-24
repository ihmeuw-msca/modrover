from itertools import combinations
from typing import Generator

from modrover.learner import LearnerID
from modrover.strategies.base import RoverStrategy


class FullExplore(RoverStrategy):

    def __init__(self, num_covariates: int):
        super().__init__(num_covariates)
        self.base_learnerid = (0,)
        self.called = False

    def generate_next_layer(self, *args, **kwargs) -> Generator:
        """Find every single possible learner ID combination, return in a single layer."""

        # Return empty generator if we've already looked for the next layer.
        # Reasoning: Fullexplore only has a single layer, so this has to be set to avoid
        # infinite looping.
        if self.called:
            yield from set()
        else:
            all_learner_ids = list(range(1, self.num_covariates + 1))
            for num_elements in range(1, self.num_covariates + 1):
                yield from map(self._as_learner_id, combinations(all_learner_ids, num_elements))
            self.called = True

    def get_upstream_learner_ids(self, learner_id: LearnerID):
        """This method is irrelevant for full explore.

        There are no dependencies, we just fit every single combination of covariates."""
        raise NotImplementedError
