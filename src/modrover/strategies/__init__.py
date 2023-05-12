from modrover.strategies.backward import Backward
from modrover.strategies.forward import Forward
from modrover.strategies.full import Full


def get_strategy(strategy_name: str):
    strategy_type_dict = {
        "full": Full,
        "forward": Forward,
        "backward": Backward,
    }
    try:
        return strategy_type_dict[strategy_name]
    except KeyError:
        raise ValueError(
            f"{strategy_name} is not a recognized strategy name. "
            f"Please select one of {list(strategy_type_dict.keys())}"
        )
