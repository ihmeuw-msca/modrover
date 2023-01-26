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

    def get_next_layer(
        self,
        curr_layer: set[LearnerID],
        learners: dict[LearnerID, Learner],
        **kwargs
    ) -> set[LearnerID]:
        """Find every single possible learner ID combination, return in a
        single layer.
        """
        if curr_layer == self.first_layer:
            return self.second_layer
        if curr_layer == self.second_layer:
            return set()
        raise ValueError(
            "curr_layer can only be the set of base_learner_id or "
            "the entire rest of the learner ids."
        )

    def _get_upstream_learner_ids(
        self,
        learner_id: LearnerID,
        learners: dict[LearnerID, Learner],
    ) -> set[LearnerID]:
        if learner_id == self.base_learner_id:
            return set()
        if learner_id in self.second_layer:
            return self.first_layer & set(learners.keys())
        raise ValueError("unrecognized learner_id")
