from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict, Generator, Iterable, Set

from .learnerid import LearnerID
from .learner import Learner


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
            performances: Dict[LearnerID, Learner]) -> Iterable:
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

    def _filter_cov_ids_set(self,
                            current_learner_ids: set[LearnerID],
                            performances: dict[LearnerID, Learner],
                            threshold: float = 1.0,
                            num_best: int = 1) -> set[LearnerID]:
        """Filter out low-performing covariate ids from selection.

        Algorithm:
        Select the n best performing covariate ids out of the input set
        Drop if this candidate model performed worse than any of its parent
        Return the remainder
        """
        sorted_learner_ids = sorted(current_learner_ids,
                                    key=lambda x: performances[x].performance)
        # Select the n best
        best_learner_ids = set(sorted_learner_ids[-num_best:])

        # Compare to the comparison layer.
        for learner_id in best_learner_ids:
            # If any upstream has a performance exceeding the current, don't explore the
            # downstream ids.
            upstreams = self.get_upstream_learner_ids(learner_id)
            current_performance = performances[learner_id].performance
            for upstream_learner_id in upstreams:
                try:
                    previous_performance = performances[upstream_learner_id].performance
                except KeyError:
                    continue
                if current_performance / previous_performance < threshold:
                    # Remove the current id from consideration
                    best_learner_ids.remove(learner_id)
                    break

        return best_learner_ids


class FullExplore(RoverStrategy):

    def __init__(self, num_covariates: int):
        super().__init__(num_covariates)

    def generate_next_layer(self, *args, **kwargs) -> Generator:
        """Find every single possible learner ID combination, return in a single layer."""
        all_learner_ids = list(range(self.num_covariates + 1))
        for num_elements in range(1, self.num_covariates + 1):
            yield from combinations(all_learner_ids, num_elements)

    def get_upstream_learner_ids(self, learner_id: LearnerID):
        """This method is irrelevant for full explore.

        There are no dependencies, we just fit every single combination of covariates."""
        raise NotImplementedError


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
        remaining_cov_ids = self._filter_cov_ids_set(current_learner_ids)
        for learner_id in remaining_cov_ids:
            candidate_ids = learner_id.create_children(self.num_covariates)
            next_learner_ids |= candidate_ids
        return next_learner_ids

    def get_upstream_learner_ids(self, learner_id: LearnerID) -> set[LearnerID]:
        """Return the possible previous nodes.

        For DownExplore, this is the current learner id's parents."""
        return set(learner_id.create_parents())


class UpExplore(RoverStrategy):

    def __init__(self, num_covariates: int):
        super().__init__(num_covariates)
        self.base_learnerid = LearnerID(tuple(range(num_covariates + 1)))

    def generate_next_layer(
            self,
            current_learner_ids: Set[LearnerID],
            performances: Dict[LearnerID, Learner]) -> set[LearnerID]:
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


strategy_type_dict = {
    "full": FullExplore,
    "down": DownExplore,
    "up": UpExplore,
}
