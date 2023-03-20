import numpy as np
import pandas as pd
from modrover.learner import Learner
from modrover.rover import Rover


def test_rover():
    data = np.random.randn(25, 3)
    columns = ["var_a", "var_b", "y"]
    dataframe = pd.DataFrame(data, columns=columns)
    # Fill in intercept and holdout columns
    dataframe["intercept"] = 1
    dataframe["holdout"] = np.random.randint(0, 2, 25)

    rover = Rover(
        model_type="gaussian",
        obs="y",
        cov_fixed=["intercept"],
        cov_exploring=["var_a", "var_b"],
        holdouts=["holdout"],
    )
    rover.fit(data=dataframe, strategies=["full"], ratio_cutoff=0.0)
    assert set(rover.learners.keys()) == {tuple(), (0,), (1,), (0, 1)}
    assert isinstance(rover.super_learner, Learner)


def test_two_parameter_rover():
    data = np.random.randn(25, 5)
    columns = ["var_a", "var_b", "var_c", "var_d", "y"]
    dataframe = pd.DataFrame(data, columns=columns)
    # Fill in intercept and holdout columns
    dataframe["intercept"] = 1
    dataframe["holdout"] = np.random.randint(0, 2, 25)

    rover = Rover(
        model_type="tobit",
        obs="y",
        cov_fixed=["intercept", "var_a"],
        cov_exploring=["var_c", "var_d"],
        main_param="mu",
        param_specs={"sigma": {"variables": ["intercept", "var_b"]}},
        holdouts=["holdout"],
    )

    rover.fit(data=dataframe, strategies=["backward"], ratio_cutoff=0.0)
    assert set(rover.learners.keys()) == {tuple(), (0,), (1,), (0, 1)}
    # Should have 6 coefficients:
    #   mu - intercept, a, c, d
    #   sigma - intercept, b
    assert rover.super_learner.coef.shape == (6,)
