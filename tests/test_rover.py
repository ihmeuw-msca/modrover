from modrover.learner import Learner
from modrover.learnerid import LearnerID
from modrover.rover import Rover


def test_get_learner():
    rover = Rover(
        model_type="gaussian",
        col_obs="obs",
        col_weights="weights",
        col_fixed={"mu": ["intercept"]},
        col_covs={"mu": ["cov1", "cov2", "cov3"]},
        col_offset={"mu": "offset"}
    )

    learner = rover.get_learner(LearnerID((0, 1, 2)))

    assert isinstance(learner, Learner)
