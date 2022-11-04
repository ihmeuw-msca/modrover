from __future__ import annotations

from typing import Optional, Tuple


class ModelID:

    def __init__(self, cov_ids: Tuple[int, ...], num_covs: Optional[int] = None) -> None:
        self.cov_ids: Tuple[int, ...]
        self.num_covs: int
        self.cov_ids, self.num_covs = self._validate_covariate_ids(cov_ids, num_covs)

    def _validate_covariate_ids(
            self, cov_ids: Tuple[int, ...], num_covs: Optional[int] = None
    ) -> Tuple[Tuple[int, ...], int]:
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

        if num_covs is None:
            num_covs = max(cov_ids)

        num_covs = max(int(num_covs), 0)

        if not all(map(lambda x: 0 <= x <= num_covs, cov_ids)):
            raise ValueError(f"covariate index is out of bounds")

        if 0 not in cov_ids:
            # Intercept always a fixed covariate, present in all models
            cov_ids.insert(0, 0)

        return tuple(cov_ids), num_covs

    @classmethod
    def _create_modelid(
            cls, cov_ids: Tuple[int], num_covs: Optional[int] = None
    ) -> ModelID:
        """
        Create a ModelID instance given a set of covariate ids and a total number of covariates

        :param cov_ids: Tuple(int)
        :param num_covs: int
        :return: ModelID set
        """
        return cls(cov_ids, num_covs)

    @property
    def children(self) -> list["ModelID"]:
        children = [
            self._create_modelid(
                cov_ids=(*self.cov_ids, i),
                num_covs=self.num_covs
            )
            for i in range(1, self.num_covs + 1)
            if i not in self.cov_ids
        ]
        return children

    @property
    def parents(self) -> list["ModelID"]:
        parents = [
            self._create_modelid(
                cov_ids=(*self.cov_ids[:i], *self.cov_ids[(i + 1):]),
                num_covs=self.num_covs
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