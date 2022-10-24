from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, Type

from .globals import metric_dict, model_type_dict


@dataclass
class ModelSpecs:

    col_id: Tuple[str, ...]
    col_obs: str
    col_fixed_covs: Tuple[str, ...]
    col_covs: Tuple[str, ...]
    col_holdout: Tuple[str, ...]
    col_offset: str = "offset"
    col_weights: str = "weights"
    col_eval_obs: str = "obs"
    col_eval_pred: str = "pred"
    model_type: Type = model_type_dict["gaussian"]
    optimizer_options: Dict = field(default_factory=dict)
    model_param_name: str = field(init=False)

    def __post_init__(self):
        if isinstance(self.model_type, str):
            self.model_type = model_type_dict[self.model_type]
        self.model_param_name = self.model_type.param_names[0]

    @property
    def all_covs(self) -> Tuple[str, ...]:
        return (*self.col_fixed_covs, *self.col_covs)

    @property
    def col_kept(self) -> Tuple[str, ...]:
        col_kept = set((
            *self.col_id,
            *self.col_holdout,
            self.col_eval_obs,
            self.col_eval_pred
        ))
        return list(col_kept)


@dataclass
class ModelEval:

    metric: Callable = metric_dict["r2"]

    def __post_init__(self):
        if isinstance(self.metric, str):
            self.metric = metric_dict[self.metric]


@dataclass
class RoverSpecs:

    strategy_names: Tuple[str, ...] = ("down",)
    strategy_options: Dict = field(default_factory=dict)


@dataclass
class SynthSpecs:

    cov_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metric_type: str = "outsample"
    max_num_models: int = 10
    kernel_param: float = 10.0
    ratio_cutoff: float = 0.99

    def __post_init__(self):
        for key, value in self.cov_bounds.items():
            self.cov_bounds[key] = tuple(map(float, value))
