from modrover.learner import Learner, LearnerID
from modrover.strategies.base import RoverStrategy


class ForwardExplore(RoverStrategy):

    @property
    def base_learner_id(self) -> LearnerID:
        return (0,)

    def get_next_layer(
        self,
        curr_layer: set[LearnerID],
        learners: dict[LearnerID, Learner],
        min_improvement: float = 1.0,
        max_len: int = 1,
    ) -> set[LearnerID]:
        """
        The down strategy will select a set of learner IDs numbering one more than the current.

        E.g. if the full set of ids is 1-5, and our current is (0,1)

        The children will be (0,1,2), (0,1,3), (0,1,4), (0,1,5)

        :param curr_layer: the current layer of learner IDs
        :param learners: dictionary storing prior scored models
        :param min_improvement: learners must out-perform parents by this ratio to continue exploring
        :param max_len: the number of best learner IDs in this layer to propagate
        :return:
        """
        learner_ids = self._filter_curr_layer(
            curr_layer=curr_layer,
            learners=learners,
            min_improvement=min_improvement,
            max_len=max_len
        )
        next_layer = set()
        for learner_id in learner_ids:
            next_layer |= self._get_learner_id_children(learner_id)
        return next_layer

    def _get_upstream_learner_ids(
        self,
        learner_id: LearnerID,
        learners: dict[LearnerID, Learner],
    ) -> set[LearnerID]:
        """Return the possible previous nodes.
        For ForwardExplore, this is the current learner id's parents.
        """
        return self._get_learner_id_parents(learner_id) & set(learners.keys())
