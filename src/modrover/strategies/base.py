from abc import ABC, abstractmethod
from typing import Iterable

from modrover.learner import Learner, LearnerID


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
            current_learner_ids: set[LearnerID],
            prior_learners: dict[LearnerID, Learner]) -> Iterable:
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

    def _as_learner_id(self, cov_ids: tuple[int, ...]) -> LearnerID:
        """
        Validate the provided covariate_id set by the number of total covariates.

        :param cov_ids: Iterable of integer cov_ids
        :param num_covs: Total number of covariates
        :return: Validated cov_ids and num_covs
        """
        # Deduplicate cov_ids
        cov_ids = set(cov_ids)
        # Sort the covariate ids since we need them in a fixed order for mapping later
        cov_ids = list(map(int, cov_ids))
        cov_ids.sort()

        if not all(map(lambda x: 0 <= x, cov_ids)):
            raise ValueError("Cannot have negative covariate IDs")

        if 0 not in cov_ids:
            # Intercept always a fixed covariate, present in all models
            cov_ids.insert(0, 0)

        return tuple(cov_ids)

    def _get_learner_id_children(self, learner_id: LearnerID) -> set[LearnerID]:
        """
        Create a new set of child covariate ID combinations based on the current one.
        As an example, if we have 5 total covariates 1-5, and our current covariate ID
        is (0,1,2), this will return
        [(0,1,2,3), (0,1,2,4), (0,1,2,5)]
        :param num_covs: total number of covariates represented
        :return: A list of LearnerID classes wrapping the child covariate ID tuples
        """
        all_covs_ids = set(range(1, self.num_covariates + 1))
        remaining_cov_ids = all_covs_ids - set(learner_id)
        children = {
            self._as_learner_id((*learner_id, cov_id))
            for cov_id in remaining_cov_ids}
        return children

    def _get_learner_id_parents(self, learner_id: LearnerID) -> set[LearnerID]:
        """
        Create a parent LearnerID class with one less covariate than the current modelid.
        As an example, if our current covariate_id tuple is (0,1,2),
        this function will return [(0,1), (0,2)]
        :return:
        """
        parents = {
            self._as_learner_id((*learner_id[:i], *learner_id[(i + 1):]))
            for i in range(1, len(learner_id))
        }
        return parents

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
