from abc import ABC, abstractmethod

from modrover.learner import Learner, LearnerID


class RoverStrategy(ABC):
    """
    An abstract base class representing a valid Rover strategy.

    The strategy is responsible for selecting the next set of LearnerIDs, determining the next
    layer of individual learners that Rover will fit. Fitting learners and storing results
    is managed by Rover itself.

    """

    def __init__(self, num_covs: int) -> None:
        self.num_covs = num_covs

    @property
    @abstractmethod
    def base_learner_id(self) -> LearnerID:
        """Starting learner id of the strategy"""

    @abstractmethod
    def get_next_layer(
        self,
        curr_layer: set[LearnerID],
        learners: dict[LearnerID, Learner],
        **kwargs,
    ) -> set[LearnerID]:
        """Abstract method to generate the next set of learner IDs."""

    @abstractmethod
    def _get_upstream_learner_ids(
        self,
        learner_id: LearnerID,
        learners: dict[LearnerID, Learner],
    ) -> set[LearnerID]:
        """Given a learnerID, generate the upstream nodes.

        Regardless of the selected strategy, we should be able to guarantee at least one
        of the upstreams has already been visited if we are at a particular node.

        Note that this is going to search the opposite direction of the specified strategy.
        e.g. for DownExplore, the upstream IDs are going to be parents
        """

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

        return tuple(cov_ids)

    def _get_learner_id_children(self, learner_id: LearnerID) -> set[LearnerID]:
        """
        Create a new set of child covariate ID combinations based on the current one.
        As an example, if we have 5 total explore covariates 0-4, and our current covariate ID
        is (0,1,2), this will return
        [(0,1,2,3), (0,1,2,4)]
        :param num_covs: total number of covariates represented
        :return: A list of LearnerID classes wrapping the child covariate ID tuples
        """
        all_covs_ids = set(range(self.num_covs))
        remaining_cov_ids = all_covs_ids - set(learner_id)
        children = {
            self._as_learner_id((*learner_id, cov_id)) for cov_id in remaining_cov_ids
        }
        return children

    def _get_learner_id_parents(self, learner_id: LearnerID) -> set[LearnerID]:
        """
        Create a parent LearnerID class with one less covariate than the current modelid.
        As an example, if our current covariate_id tuple is (0,1,2),
        this function will return [(0,1), (0,2), (1, 2)]
        :return:
        """
        parents = {
            self._as_learner_id((*learner_id[:i], *learner_id[(i + 1) :]))
            for i in range(len(learner_id))
        }
        return parents

    def _filter_curr_layer(
        self,
        curr_layer: set[LearnerID],
        learners: dict[LearnerID, Learner],
        min_improvement: float = 1.0,
        max_len: int = 1,
    ) -> set[LearnerID]:
        """Filter out low-performing covariate ids from selection.

        Algorithm:
        Select the n best performing covariate ids out of the input set
        Drop if this candidate model performed worse than any of its parent
        Return the remainder
        """
        sorted_learner_ids = sorted(
            curr_layer, key=lambda learner_id: learners[learner_id].score
        )
        # Select the best max_len learner ids
        learner_ids = set(sorted_learner_ids[-max_len:])

        # Compare to the comparison layer.
        learner_ids_to_remove = set()
        for learner_id in learner_ids:
            upstream_learner_ids = self._get_upstream_learner_ids(
                learner_id,
                learners,
            )
            curr_score = learners[learner_id].score
            for upstream_learner_id in upstream_learner_ids:
                prev_score = learners[upstream_learner_id].score
                if curr_score / prev_score < min_improvement:
                    learner_ids_to_remove.add(learner_id)
                    break

        return learner_ids - learner_ids_to_remove
