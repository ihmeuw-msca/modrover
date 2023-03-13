from typing import Optional

import numpy as np
import pytest

from modrover.learner import Learner, LearnerID
from modrover.rover import Rover


class MockLearner(Learner):
    """Mock learner that comes 'prefit' with coefficients."""

    def __init__(self, coefficients: np.ndarray, performance: float):
        self.performance = performance
        self.model = (
            lambda: None
        )  # arbitrary mutable object that we can assign weights to
        self.model.size = len(coefficients)

        self.coef = coefficients


class MockRover(Rover):
    """Mock Rover that comes 'prefit' with some mock data."""

    def __init__(
        self,
        learners: dict[LearnerID, Learner],
        main_param: str,
        cov_fixed: list[str],
        cov_exploring: list[str],
        param_specs: Optional[dict[str, dict]] = None,
        model_type: str = "gaussian",
    ):
        self.model_type = self._as_model_type(model_type)
        self.main_param = self._as_main_param(main_param)
        self.cov_fixed, self.cov_exploring = self._as_cov(cov_fixed, cov_exploring)
        self.param_specs = self._as_param_specs(param_specs)

        self.learners = learners


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
        main_param="mu",
        cov_fixed=[0],
        cov_exploring=[1, 2, 3, 4],
    )
    return rover
