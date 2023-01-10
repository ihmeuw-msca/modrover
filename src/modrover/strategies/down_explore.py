from typing import Dict, Set

from modrover.learner import Learner
from modrover.learnerid import LearnerID
from modrover.strategies.base import RoverStrategy


class DownExplore(RoverStrategy):

    def __init__(self, num_covariates: int):
        super().__init__(num_covariates)
        self.base_learnerid = LearnerID((0,))

    def generate_next_layer(
            self,
            current_learner_ids: Set[LearnerID],
            performances: Dict[LearnerID, Learner]) -> set[LearnerID]:
        """
        The down strategy will select a set of learner IDs numbering one more than the current.

        E.g. if the full set of ids is 1-5, and our current is (0,1)

        The children will be (0,1,2), (0,1,3), (0,1,4), (0,1,5)

        :param current_learner_ids: the current layer of learner IDs
        :param all_learner_ids:
        :param performances:
        :return:
        """
        next_learner_ids = set()
        remaining_cov_ids = self._filter_learner_ids(
            current_learner_ids=current_learner_ids,
            performances=performances,
        )
        for learner_id in remaining_cov_ids:
            candidate_ids = learner_id.create_children(self.num_covariates)
            next_learner_ids |= candidate_ids
        return next_learner_ids

    def get_upstream_learner_ids(self, learner_id: LearnerID) -> set[LearnerID]:
        """Return the possible previous nodes.

        For DownExplore, this is the current learner id's parents."""
        return set(learner_id.create_parents())
