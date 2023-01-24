from modrover.learner import Learner, LearnerID
from modrover.strategies.base import RoverStrategy


class ForwardExplore(RoverStrategy):

    def __init__(self, num_covariates: int):
        super().__init__(num_covariates)
        self.base_learner_id = LearnerID((0,))

    def generate_next_layer(
            self,
            current_learner_ids: set[LearnerID],
            prior_learners: dict[LearnerID, Learner],
            threshold: float = 1.0,
            num_best: int = 1,
    ) -> set[LearnerID]:
        """
        The down strategy will select a set of learner IDs numbering one more than the current.

        E.g. if the full set of ids is 1-5, and our current is (0,1)

        The children will be (0,1,2), (0,1,3), (0,1,4), (0,1,5)

        :param current_learner_ids: the current layer of learner IDs
        :param prior_learners: dictionary storing prior scored models
        :param threshold: learners must out-perform parents by this ratio to continue exploring
        :param num_best: the number of best learner IDs in this layer to propagate
        :return:
        """
        next_learner_ids = set()
        remaining_cov_ids = self._filter_learner_ids(
            current_learner_ids=current_learner_ids,
            prior_learners=prior_learners,
            threshold=threshold,
            num_best=num_best
        )
        for learner_id in remaining_cov_ids:
            candidate_ids = self._get_learner_id_children(learner_id)
            next_learner_ids |= set(candidate_ids)
        return next_learner_ids

    def get_upstream_learner_ids(self, learner_id: LearnerID) -> set[LearnerID]:
        """Return the possible previous nodes.

        For DownExplore, this is the current learner id's parents."""
        return self._get_learner_id_parents(learner_id)
