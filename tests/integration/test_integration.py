import numpy as np
import pandas as pd

from modrover.learner import Learner
from modrover.rover import Rover


def test_rover():

    data = np.random.randn(25, 3)
    columns = [
        'var_a',
        'var_b',
        'y']
    dataframe = pd.DataFrame(data, columns=columns)
    # Fill in intercept and holdout columns
    dataframe['intercept'] = 1
    dataframe['holdout'] = np.random.randint(0, 2, 25)

    rover = Rover(
        model_type='gaussian',
        y='y',
        col_fixed={'mu': ['intercept']},
        col_explore=['var_a', 'var_b'],
        explore_param='mu',
        holdout_cols=['holdout']
    )

    rover.fit(dataset=dataframe, strategy='full', ratio_cutoff=.0)
    assert set(rover.learners.keys()) == {tuple(), (0,), (1,), (0, 1)}
    assert isinstance(rover.super_learner, Learner)
