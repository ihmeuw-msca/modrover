from modrover.learner import Learner, LearnerID
from modrover.strategies.base import RoverStrategy


class ForwardExplore(RoverStrategy):
    """Forward strategy starts from the model with only the :code:`cov_fixed`
    and explores forwards with more and more covariates base on the learner
    performance.

    """

    @property
    def base_learner_id(self) -> LearnerID:
        return tuple()

    def get_next_layer(
        self,
        curr_layer: set[LearnerID],
        learners: dict[LearnerID, Learner],
        min_improvement: float = 1.0,
        max_len: int = 1,
    ) -> set[LearnerID]:
        """
        The forward strategy will select a set of learner IDs numbering one more
        than the current. This function will return next set of learner ids
        corresponding the learers that need to be fitted.

        E.g. if the full set of ids is 1-5, and our current is (0, 1)

        The children will be (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5)

        Parameters
        ----------
        min_improvement
            Minimum performance improvement requirement for the learner to be
            consider qualified to generate the next layer.
        max_len
            Maximum number of learner in each layer.

        """
        learner_ids = self._filter_curr_layer(
            curr_layer=curr_layer,
            learners=learners,
            min_improvement=min_improvement,
            max_len=max_len,
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
