from itertools import combinations

from modrover.learner import Learner, LearnerID
from modrover.strategies.base import RoverStrategy


class FullExplore(RoverStrategy):
    """Full strategy explore every possible covariate combinations."""

    @property
    def base_learner_id(self) -> LearnerID:
        return tuple()

    @property
    def first_layer(self) -> set[LearnerID]:
        """Leanrer ids corresponding to the first layer."""
        return {self.base_learner_id}

    @property
    def second_layer(self) -> set[LearnerID]:
        """Leanrer ids corresponding to the second layer."""
        all_cov_ids = range(self.num_covs)
        second_layer = []
        for num_elements in range(1, self.num_covs + 1):
            second_layer.extend(
                map(self._as_learner_id, combinations(all_cov_ids, num_elements))
            )
        return set(second_layer)

    def get_next_layer(
        self, curr_layer: set[LearnerID], learners: dict[LearnerID, Learner], **kwargs
    ) -> set[LearnerID]:
        """Find every single possible learner ID combination, return in a
        single layer. If the :code:`curr_layer=first_layer`, it will return the
        :code:`second_layer` and if the :code:`curr_layer=second_layer`, it will
        return an empty layer. The first layer and the second layer together
        cover all possible combinations.

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
