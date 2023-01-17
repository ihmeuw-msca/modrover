from abc import ABC, abstractmethod
from typing import Dict, Iterable, Set

from modrover.learnerid import LearnerID
from modrover.learner import Learner


class RoverStrategy(ABC):
    """
    An abstract base class representing a valid Rover strategy.

    The strategy is responsible for selecting the next set of LearnerIDs, determining the next
    layer of individual learners that Rover will fit. Fitting learners and storing results
    is managed by Rover itself.

    """

    @abstractmethod
    def __init__(self, num_covariates: int):
        self.num_covariates = num_covariates
        self.base_learnerid: LearnerID

    @abstractmethod
    def generate_next_layer(
            self,
            current_learner_ids: Set[LearnerID],
            prior_learners: Dict[LearnerID, Learner]) -> Iterable:
        """Abstract method to generate the next set of learner IDs."""
        raise NotImplementedError

    @abstractmethod
    def get_upstream_learner_ids(self, learner_id: LearnerID):
        """Given a learnerID, generate the upstream nodes.

        Regardless of the selected strategy, we should be able to guarantee at least one
        of the upstreams has already been visited if we are at a particular node.

        Note that this is going to search the opposite direction of the specified strategy.
        e.g. for DownExplore, the upstream IDs are going to be parents
        """
        raise NotImplementedError

    def _filter_learner_ids(
            self,
            current_learner_ids: set[LearnerID],
            prior_learners: dict[LearnerID, Learner],
            threshold: float = 1.0,
            num_best: int = 1) -> set[LearnerID]:
        """Filter out low-performing covariate ids from selection.

        Algorithm:
        Select the n best performing covariate ids out of the input set
        Drop if this candidate model performed worse than any of its parent
        Return the remainder
        """
        sorted_learner_ids = sorted(current_learner_ids,
                                    key=lambda x: prior_learners[x].performance)
        # Select the n best
        best_learner_ids = set(sorted_learner_ids[-num_best:])

        # Compare to the comparison layer.
        learner_ids_to_remove = set()
        for learner_id in best_learner_ids:
            # If any upstream has a performance exceeding the current, don't explore the
            # downstream ids.
            upstreams = self.get_upstream_learner_ids(learner_id)
            current_performance = prior_learners[learner_id].performance
            for upstream_learner_id in upstreams:
                if upstream_learner_id in prior_learners:
                    previous_performance = prior_learners[upstream_learner_id].performance
                    if current_performance / previous_performance < threshold:
                        # Remove the current id from consideration
                        learner_ids_to_remove.add(learner_id)
                        break

        return best_learner_ids - learner_ids_to_remove
