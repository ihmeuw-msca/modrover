from modrover.strategies.backward_explore import BackwardExplore
from modrover.strategies.forward_explore import ForwardExplore
from modrover.strategies.full_explore import FullExplore


def get_strategy(strategy_name: str):
    strategy_type_dict = {
        "full": FullExplore,
        "forward": ForwardExplore,
        "backward": BackwardExplore,
    }
    try:
        return strategy_type_dict[strategy_name]
    except KeyError:
        raise ValueError(
            f"{strategy_name} is not a recognized strategy name. "
            f"Please select one of {list(strategy_type_dict.keys())}"
        )
