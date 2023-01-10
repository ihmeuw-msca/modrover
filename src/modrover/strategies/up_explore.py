from modrover.learner import Learner
from modrover.learnerid import LearnerID
from modrover.strategies.base import RoverStrategy


class UpExplore(RoverStrategy):

    def __init__(self, num_covariates: int):
        super().__init__(num_covariates)
        self.base_learnerid = LearnerID(tuple(range(num_covariates + 1)))

    def generate_next_layer(
            self,
            current_learner_ids: set[LearnerID],
            performances: dict[LearnerID, Learner]) -> set[LearnerID]:
        """
        The down strategy will select a set of learner IDs numbering one more than the current.

        E.g. if the full set of ids is 1-5, and our current is (0,1)

        The children will be (0,1,2), (0,1,3), (0,1,4), (0,1,5)

        :param current_learner_ids:
        :param all_learner_ids:
        :param performances:
        :return:
        """
        next_learner_ids = set()
        remaining_cov_ids = self._filter_cov_ids_set(
            current_learner_ids=current_learner_ids,
            performances=performances,
        )
        for learner_id in remaining_cov_ids:
            candidate_ids = learner_id.create_parents()
            next_learner_ids |= candidate_ids
        return next_learner_ids

    def get_upstream_learner_ids(self, learner_id: LearnerID):
        """Return the possible previous nodes.

        For UpExplore, this is the current learner id's children."""
        return learner_id.create_children(self.num_covariates)