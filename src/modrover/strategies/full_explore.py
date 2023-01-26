from itertools import combinations

from modrover.learner import LearnerID
from modrover.strategies.base import RoverStrategy


class FullExplore(RoverStrategy):

    def __init__(self, num_covs: int) -> None:
        super().__init__(num_covs)
        self.base_learner_id = (0,)
        self.called = False

    def generate_next_layer(self, *args, **kwargs) -> set[LearnerID]:
        """Find every single possible learner ID combination, return in a single layer."""

        # Return empty generator if we've already looked for the next layer.
        # Reasoning: Fullexplore only has a single layer, so this has to be set to avoid
        # infinite looping.

        if self.called:
            return set()

        all_cov_ids = range(1, self.num_covs + 1)

        remaining_learner_ids = []
        for num_elements in range(1, self.num_covs + 1):
            remaining_learner_ids.extend(map(
                self._as_learner_id, combinations(all_cov_ids, num_elements)
            ))
        self.called = True
        return set(remaining_learner_ids)

    def get_upstream_learner_ids(self, learner_id: LearnerID):
        """This method is irrelevant for full explore.

        There are no dependencies, we just fit every single combination of covariates."""
        raise NotImplementedError
