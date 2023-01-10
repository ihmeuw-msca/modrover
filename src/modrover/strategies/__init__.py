from modrover.strategies.down_explore import DownExplore
from modrover.strategies.full_explore import FullExplore
from modrover.strategies.up_explore import UpExplore

strategy_type_dict = {
    "full": FullExplore,
    "down": DownExplore,
    "up": UpExplore,
}
