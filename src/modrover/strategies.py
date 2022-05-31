from abc import ABC, abstractmethod

from .modelhub import ModelHub
from .types import CovIDs, CovIDsSet


class RoverStrategy(ABC):

    def __init__(self, modelhub: ModelHub):
        self.modelhub = modelhub
        self.performances = {
            cov_ids: self.modelhub.get_model_performance(cov_ids)
            for cov_ids in self.modelhub.get_ran_cov_ids_set()
        }

    @abstractmethod
    def implement(self, **kwargs):
        pass

    def run_model(self, cov_ids: CovIDs):
        self.modelhub.run_model(cov_ids)
        if cov_ids not in self.performances and self.modelhub._has_ran(cov_ids):
            performance = self.modelhub.get_model_performance(cov_ids)
            self.performances[cov_ids] = performance


class FullExplore(RoverStrategy):

    def implement(self, **kwargs):
        cov_ids_set = self.modelhub.get_full_cov_ids_set()
        for cov_ids in cov_ids_set:
            self.modelhub.run_model(cov_ids)


class DownExplore(RoverStrategy):

    def _filter_cov_ids_set(self,
                            cov_ids_set: CovIDsSet,
                            threshold: float = 1.0,
                            level_lb: int = 1,
                            level_ub: int = 1) -> CovIDsSet:
        # remove the the covariate id that doesn't have performance metric
        ran_cov_ids_set = set(self.performances.keys())
        cov_ids_set = (
            cov_ids_set - (cov_ids_set - ran_cov_ids_set)
        )
        if len(cov_ids_set) <= level_lb:
            return cov_ids_set
        # rank covariate ids by their performance
        cov_ids_list = list(cov_ids_set)
        cov_ids_list.sort(key=lambda x: self.performances[x])
        cov_ids_list = cov_ids_list[-level_ub:]
        cov_ids_set = set(cov_ids_list)
        for cov_ids in cov_ids_list:
            if len(cov_ids_set) <= level_lb:
                break
            compare_cov_ids_set = self.modelhub.get_parent_cov_ids_set(
                cov_ids, ran_cov_ids_set
            )
            if any(
                self.performances[cov_ids] / self.performances[compare_cov_ids] < threshold
                for compare_cov_ids in compare_cov_ids_set
            ):
                cov_ids_set.remove(cov_ids)
        return cov_ids_set

    def implement(self, **kwargs):
        # fit root model
        self.modelhub.run_model(tuple())
        # fit first layer
        next_cov_ids_set = self.modelhub.get_child_cov_ids_set(tuple())
        for cov_ids in next_cov_ids_set:
            self.run_model(cov_ids)

        # filter the next layer
        curr_cov_ids_set = self._filter_cov_ids_set(
            next_cov_ids_set, **kwargs
        )
        next_cov_ids_set = set().union(*[
            self.modelhub.get_child_cov_ids_set(cov_ids)
            for cov_ids in curr_cov_ids_set
        ])
        while len(next_cov_ids_set) > 0:
            for cov_ids in next_cov_ids_set:
                self.run_model(cov_ids)
            curr_cov_ids_set = self._filter_cov_ids_set(
                next_cov_ids_set, **kwargs
            )
            next_cov_ids_set = set().union(*[
                self.modelhub.get_child_cov_ids_set(cov_ids)
                for cov_ids in curr_cov_ids_set
            ])


class UpExplore(RoverStrategy):

    def _filter_cov_ids_set(self,
                            cov_ids_set: CovIDsSet,
                            threshold: float = 1.0,
                            level_lb: int = 1,
                            level_ub: int = 1) -> CovIDsSet:
        # remove the the covariate id that doesn't have performance metric
        ran_cov_ids_set = set(self.performances.keys())
        cov_ids_set = (
            cov_ids_set - (cov_ids_set - ran_cov_ids_set)
        )
        if len(cov_ids_set) <= level_lb:
            return cov_ids_set
        # rank covariate ids by their performance
        cov_ids_list = list(cov_ids_set)
        cov_ids_list.sort(key=lambda x: self.performances[x])
        cov_ids_list = cov_ids_list[-level_ub:]
        cov_ids_set = set(cov_ids_list)
        for cov_ids in cov_ids_list:
            if len(cov_ids_set) <= level_lb:
                break
            compare_cov_ids_set = self.modelhub.get_child_cov_ids_set(
                cov_ids, ran_cov_ids_set
            )
            if any(
                self.performances[cov_ids] / self.performances[compare_cov_ids] < threshold
                for compare_cov_ids in compare_cov_ids_set
            ):
                cov_ids_set.remove(cov_ids)
        return cov_ids_set

    def implement(self, **kwargs):
        # fit full model
        full_cov_ids = self.modelhub.get_full_cov_ids()
        self.modelhub.run_model(full_cov_ids)
        # fit first layer
        next_cov_ids_set = self.modelhub.get_parent_cov_ids_set(full_cov_ids)
        for cov_ids in next_cov_ids_set:
            self.run_model(cov_ids)

        # filter the next layer
        curr_cov_ids_set = self._filter_cov_ids_set(
            next_cov_ids_set, **kwargs
        )
        next_cov_ids_set = set().union(*[
            self.modelhub.get_parent_cov_ids_set(cov_ids)
            for cov_ids in curr_cov_ids_set
        ])
        while len(next_cov_ids_set) > 0:
            for cov_ids in next_cov_ids_set:
                self.run_model(cov_ids)
            curr_cov_ids_set = self._filter_cov_ids_set(
                next_cov_ids_set, **kwargs
            )
            next_cov_ids_set = set().union(*[
                self.modelhub.get_parent_cov_ids_set(cov_ids)
                for cov_ids in curr_cov_ids_set
            ])


strategy_type_dict = {
    "full": FullExplore,
    "down": DownExplore,
    "up": UpExplore,
}
