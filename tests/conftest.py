from typing import Optional

import numpy as np
import pytest

from modrover.learner import Learner
from modrover.rover import Rover


class MockLearner(Learner):
    """Mock learner that comes 'prefit' with coefficients."""

    def __init__(self, coefficients: np.ndarray, performance: float):
        self.performance = performance
        self._model = (
            lambda: None
        )  # arbitrary mutable object that we can assign weights to

        self.coef = coefficients


class MockRover(Rover):
    """Mock Rover that comes 'prefit' with some mock data."""

    def __init__(
        self,
        learners: dict,
        param: str,
        cov_fixed: dict,
        cov_exploring: list,
        param_specs: Optional[dict[str, dict]] = None,
        model_type: str = "gaussian",
    ):
        self.learners = learners
        self.param = param
        self.cov_fixed = cov_fixed
        self.cov_exploring = cov_exploring
        self.param_specs = param_specs
        self.model_type = model_type


@pytest.fixture
def mock_rover():
    learner_parameters = [
        ((1, 3), np.array([0.2, 0.4, 0.6]), 1.2),
        ((2, 3), np.array([0.0, 0.1, 0.5]), 1.0),
        ((3,), np.array([1.0, -0.2]), -0.3),
    ]

    learners = {
        learner_id: MockLearner(*params) for learner_id, *params in learner_parameters
    }

    rover = MockRover(
        learners=learners,
        param="mu",
        cov_fixed=[0],
        cov_exploring=[1, 2, 3, 4],
    )
    return rover
