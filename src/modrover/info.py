from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, Type

from .globals import metric_dict, model_type_dict


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
