from modrover.strategies.backward_explore import BackwardExplore
from modrover.strategies.forward_explore import ForwardExplore
from modrover.strategies.full_explore import FullExplore


strategy_type_dict = {
    "full": FullExplore,
    "forward": ForwardExplore,
    "backward": BackwardExplore,
}
