from modrover.learner import Learner, LearnerID
from modrover.rover import Rover


def test_get_learner():
    rover = Rover(
        model_type="gaussian",
        y="obs",
        col_fixed={"mu": ["intercept"]},
        col_explore=["cov1", "cov2", "cov3"],
        explore_param='mu'
    )

    learner = rover.get_learner((0, 1, 2))

    assert isinstance(learner, Learner)
    assert len(learner.all_covariates) == 4
    assert len(learner.param_specs['mu']['variables']) == 4
    # Check that the order is preserved
    assert learner.all_covariates == ['intercept', 'cov1', 'cov2', 'cov3']
    variables = learner.param_specs['mu']['variables']
    assert [var.name for var in variables] == learner.all_covariates


def test_empty_learner_id():

    # Special test for the learner with no explore covariates, only fixed
    rover = Rover(
        model_type="gaussian",
        y="obs",
        col_fixed={"mu": ["intercept"]},
        col_explore=["cov1", "cov2", "cov3"],
        explore_param='mu'
    )

    assert rover.default_param_specs == {
        'mu': {
            'variables': [
                'intercept'
            ]
        }
    }

    learner = rover.get_learner(tuple())
    variables = learner.param_specs['mu']['variables']
    assert len(variables) == 1
    assert variables[0].name == 'intercept'
