from itertools import combinations

from modrover.learner import Learner, LearnerID
from modrover.strategies.base import RoverStrategy


class FullExplore(RoverStrategy):

    @property
    def base_learner_id(self) -> LearnerID:
        return (0,)

    @property
    def first_layer(self) -> set[LearnerID]:
        return {self.base_learner_id}

    @property
    def second_layer(self) -> set[LearnerID]:
        all_cov_ids = range(1, self.num_covs + 1)
        second_layer = []
        for num_elements in range(1, self.num_covs + 1):
            second_layer.extend(map(
                self._as_learner_id, combinations(all_cov_ids, num_elements)
            ))
        return set(second_layer)

    def generate_next_layer(
        self,
        current_learner_ids: set[LearnerID],
        learners: dict[LearnerID, Learner]
    ) -> set[LearnerID]:
        """Find every single possible learner ID combination, return in a single layer."""

        # Return empty generator if we've already looked for the next layer.
        # Reasoning: Fullexplore only has a single layer, so this has to be set to avoid
        # infinite looping.

        if current_learner_ids == self.first_layer:
            return self.second_layer
        elif current_learner_ids == self.second_layer:
            return set()
        else:
            raise ValueError(
                "current_learner_ids can only be the set of base_learner_id or "
                "the entire rest of the learner ids."
            )

    def get_upstream_learner_ids(self, learner_id: LearnerID):
        """This method is irrelevant for full explore.

        There are no dependencies, we just fit every single combination of covariates."""
        raise NotImplementedError
