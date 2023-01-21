from modrover.learner import Learner, LearnerID
from modrover.rover import Rover


def test_get_learner():
    rover = Rover(
        model_type="gaussian",
        y="obs",
        col_fixed={"mu": ["intercept"]},
        col_explore={"mu": ["cov1", "cov2", "cov3"]},
    )

    learner = rover.get_learner(LearnerID((0, 1, 2)))

    assert isinstance(learner, Learner)
