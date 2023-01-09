from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict, Generator, Iterable, Set

from .learnerid import LearnerID
from .learner import Learner
from .types import CovIDs, CovIDsSet


class RoverStrategy(ABC):
    """
    An abstract base class representing a valid Rover strategy.

    The strategy is responsible for selecting the next set of LearnerIDs, determining the next
    layer of individual learners that Rover will fit. Fitting learners and storing results
    is managed by Rover itself.

    """

    @abstractmethod
    def generate_layer(self, *args, **kwargs) -> Iterable:
        raise NotImplementedError

    def _filter_cov_ids_set(self,
                            cov_ids_set: CovIDsSet,
                            performances: Dict,
                            some_callable=None,
                            threshold: float = 1.0,
                            level_lb: int = 1,
                            level_ub: int = 1) -> CovIDsSet:
        """Filter out low-performing covariate ids from selection.

        Algorithm:
        Select the n best performing covariate ids out of the input set
        Generate parents/children
        If new model has been fit,
        """
        sorted_cov_ids = sorted(cov_ids_set, lambda x: performances[x])
        # Select the n best
        best_cov_ids = sorted_cov_ids[-level_ub:]

        return best_cov_ids

class FullExplore(RoverStrategy):

    def __init__(self):
        self.base_learnerid = set()

    def generate_layer(self, all_learner_ids: Set[LearnerID], *args, **kwargs) -> Generator:
        """Find every single possible learner ID combination, return in a single layer."""
        for num_elements in range(1, len(all_learner_ids) + 1):
            yield from combinations(all_learner_ids, num_elements)


class DownExplore(RoverStrategy):

    def generate_layer(self,
                       current_learner_ids: Set[LearnerID],
                       all_learner_ids: Set[LearnerID],
                       performances: Dict[LearnerID, Learner]):
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
        remaining_cov_ids = self._filter_cov_ids_set(current_learner_ids)
        for learner_id in remaining_cov_ids:
            candidate_ids = learner_id.create_children(len(all_learner_ids))
            next_learner_ids |= candidate_ids
        return next_learner_ids


class UpExplore(RoverStrategy):

    def generate_layer(self,
                       current_learner_ids: Set[LearnerID],
                       all_learner_ids: Set[LearnerID],
                       performances: Dict[LearnerID, Learner]):
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
        remaining_cov_ids = self._filter_cov_ids_set(current_learner_ids)
        for learner_id in remaining_cov_ids:
            candidate_ids = learner_id.create_parents()
            next_learner_ids |= candidate_ids
        return next_learner_ids


strategy_type_dict = {
    "full": FullExplore,
    "down": DownExplore,
    "up": UpExplore,
}
