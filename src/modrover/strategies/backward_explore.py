from modrover.learner import Learner
from modrover.learnerid import LearnerID
from modrover.strategies.base import RoverStrategy


class BackwardExplore(RoverStrategy):

    def __init__(self, num_covariates: int):
        super().__init__(num_covariates)
        self.base_learnerid = LearnerID(tuple(range(num_covariates + 1)))

    def generate_next_layer(
            self,
            current_learner_ids: set[LearnerID],
            prior_learners: dict[LearnerID, Learner]) -> set[LearnerID]:
        """
        The backward strategy will select a set of learner IDs numbering one less.

        E.g. if the full set of ids is 1-5, and our current is (0,1, 2)

        The downstreams will be (0,1), (0,2)

        :param current_learner_ids:
        :param all_learner_ids:
        :param prior_learners:
        :return:
        """
        next_learner_ids = set()
        remaining_cov_ids = self._filter_learner_ids(
            current_learner_ids=current_learner_ids,
            prior_learners=prior_learners,
        )
        for learner_id in remaining_cov_ids:
            candidate_ids = learner_id.create_parents()
            next_learner_ids |= set(candidate_ids)
        return next_learner_ids

    def get_upstream_learner_ids(self, learner_id: LearnerID) -> set[LearnerID]:
        """Return the possible previous nodes.

        For UpExplore, this is the current learner id's children."""
        return set(learner_id.create_children(self.num_covariates))