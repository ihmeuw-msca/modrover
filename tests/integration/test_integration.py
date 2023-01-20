import numpy as np
import pandas as pd

from modrover.rover import Rover
from modrover.learnerid import LearnerID


def test_rover():

    data = np.random.randn(25, 2)
    columns = [
        'var_a',
        'y']
    dataframe = pd.DataFrame(data, columns=columns)
    # Fill in intercept and holdout columns
    dataframe['intercept'] = 1
    dataframe['holdout'] = np.random.randint(0, 2, 25)

    rover = Rover(
        model_type='gaussian',
        y='y',
        cov_fixed={'mu': ['intercept']},
        cov_explore={'mu': ['var_a']},
        holdout_cols=['holdout']
    )

    rover.explore(dataset=dataframe, strategy='full')
    assert set(rover.learners.keys()) == {(0,), (0, 1)}
