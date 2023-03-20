from modrover.learner import Learner, LearnerID
from modrover.rover import Rover


def test_get_learner():
    rover = Rover(
        model_type="gaussian",
        obs="obs",
        cov_fixed=["intercept"],
        cov_exploring=["cov1", "cov2", "cov3"],
    )

    learner = rover._get_learner((0, 1, 2))

    assert isinstance(learner, Learner)
    assert len(learner.param_specs["mu"]["variables"]) == 4
    # Check that the order is preserved
    variables = learner.param_specs["mu"]["variables"]
    assert [var.name for var in variables] == ["intercept", "cov1", "cov2", "cov3"]


def test_empty_learner_id():
    # Special test for the learner with no explore covariates, only fixed
    rover = Rover(
        model_type="gaussian",
        obs="obs",
        cov_fixed=["intercept"],
        cov_exploring=["cov1", "cov2", "cov3"],
    )

    assert rover._get_param_specs(tuple()) == {"mu": {"variables": ["intercept"]}}

    learner = rover._get_learner(tuple())
    variables = learner.param_specs["mu"]["variables"]
    assert len(variables) == 1
    assert variables[0].name == "intercept"
