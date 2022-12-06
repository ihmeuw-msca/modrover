from __future__ import annotations

from typing import Tuple


class ModelID:

    def __init__(self, cov_ids: Tuple[int, ...]) -> None:
        self.cov_ids: Tuple[int, ...]
        self.num_covs: int
        self.cov_ids = self._validate_covariate_ids(cov_ids)

    def _validate_covariate_ids(self, cov_ids: Tuple[int, ...]) -> Tuple[int, ...]:
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

    @classmethod
    def _create_modelid(cls, cov_ids: Tuple[int, ...]) -> ModelID:
        """
        Create a ModelID instance given a set of covariate ids

        :param cov_ids: Tuple(int)
        :return: ModelID set
        """
        return cls(cov_ids)

    def create_children(self, num_covs: int) -> list["ModelID"]:
        """
        Create a new set of child covariate ID combinations based on the current one.

        As an example, if we have 5 total covariates 1-5, and our current covariate ID
        is (0,1,2), this will return
        [(0,1,2,3), (0,1,2,4), (0,1,2,5)]

        :param num_covs: total number of covariates represented
        :return: A list of ModelID classes wrapping the child covariate ID tuples
        """
        children = [
            self._create_modelid(
                cov_ids=(*self.cov_ids, i),
            )
            for i in range(1, num_covs + 1)
            if i not in self.cov_ids
        ]
        return children

    def create_parents(self) -> list["ModelID"]:
        """
        Create a parent ModelID class with one less covariate than the current modelid.

        As an example, if our current covariate_id tuple is (0,1,2),
        this function will return [(0,1), (0,2)]

        :return:
        """
        parents = [
            self._create_modelid(
                cov_ids=(*self.cov_ids[:i], *self.cov_ids[(i + 1):])
            )
            for i in range(1, len(self.cov_ids))
        ]
        return parents

    def __str__(self) -> str:
        return "_".join(map(str, self.cov_ids))

    def __hash__(self) -> int:
        return hash(self.cov_ids)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)