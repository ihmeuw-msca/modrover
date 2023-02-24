from modrover.rover import Rover
from modrover.learner import Learner

import numpy as np
import pytest


class MockLearner(Learner):
    """Mock learner that comes 'prefit' with coefficients."""

    def __init__(self, coefficients: np.ndarray, performance: float):
        self.performance = performance
        self._model = lambda: None  # arbitrary mutable object that we can assign weights to

        self.opt_coefs = coefficients


class MockRover(Rover):
    """Mock Rover that comes 'prefit' with some mock data."""

    def __init__(self, learners: dict, col_fixed: dict, col_explore: dict):
        self.learners = learners
        self.col_fixed = col_fixed
        self.col_explore = col_explore


@pytest.fixture
def mock_rover():

    learner_parameters = [
        ((0, 1, 3), np.array([.2, .4, .6]), 1.2),
        ((0, 2, 3), np.array([0., .1, .5]), 1.),
        ((0, 3), np.array([1., -.2]), -.3),
    ]

    learners = {
        learner_id: MockLearner(*params)
        for learner_id, *params in learner_parameters
    }

    rover = MockRover(
        learners=learners,
        col_fixed={'mu': [0]},
        col_explore={'mu': [1, 2, 3, 4]}
    )
    return rover
