import numpy as np
import pytest

from modrover.globals import model_type_dict
from modrover.learner import Learner
from modrover.synthesizer import metrics_to_weights


@pytest.mark.parametrize(
    "metrics,num_models,kernel_param,ratio_cutoff,expectation",
    [
        # Selecting a high cutoff means only 1 value selected
        (np.array([1, 0.5, 0]), 10, 1, 0.99, np.array([1, 0, 0])),
        # Two values should be selected with same max value, equal weights
        (np.array([1, 0.5, 1]), 10, 1, 0.99, np.array([0.5, 0, 0.5])),
        # Lowering the threshold should return two values with unequal weights
        (np.array([1, 0.8, 0]), 10, 1, 0.7, np.array([0.55, 0.45, 0])),
        # Increasing the kernel parameter should bias towards the better performances
        (np.array([1, 0.8, 0]), 10, 10, 0.7, np.array([0.88, 0.12, 0])),
        # Lowering the num_models param should only return a max number of non zero weights
        (np.array([1, 0.8, 0]), 1, 1, 0.7, np.array([1, 0, 0])),
        # Negative values for performances handled gracefully
        (
            np.array([-0.5, -0.3, 0.2, 0.4]),
            10,
            1,
            -100,
            np.array([0.06, 0.09, 0.32, 0.53]),
        ),
    ],
)
def test_create_weights(metrics, num_models, kernel_param, ratio_cutoff, expectation):
    weights = metrics_to_weights(metrics, num_models, kernel_param, ratio_cutoff)
    assert np.allclose(weights, expectation, atol=0.01)
    assert np.isclose(weights.sum(), 1)


def test_get_coef_index(mock_rover):
    # Given some faux coefficients, get the correct global coefficients
    # Since learner ids represent indices of explore columns, we have the expected mapping
    #   .1 = coefficient of col 0
    #   .3 = coefficient of col 1
    #   .2 = coefficient of col 4
    learner_id = (0, 3)
    coef_index = mock_rover._get_coef_index(learner_id)
    assert np.allclose(coef_index, [0, 1, 4])


def test_two_parameter_get_coef_index(mock_rover):
    """Test that covariate aggregation works for multiple parameter models."""

    mock_rover.model_type = "tobit"
    mock_rover.param = "mu"
    mock_rover.cov_fixed = [0, 3]
    mock_rover.cov_exploring = [1, 2]
    mock_rover.param_specs = mock_rover._as_param_specs({"sigma": {"variables": [4]}})

    # Expected ordering out of a particular model:
    # Say learner_id = (0,) - represents both parameters
    # Expected covariate order is then [0, 3, 1, 4] - Tobit model defines mu -> sigma
    # as the parameter order
    # Rover appends the explore columns to the end of the mu fixed columns

    learner_id = (0,)
    coef_index = mock_rover._get_coef_index(learner_id)
    assert np.allclose(coef_index, [0, 1, 2, 4])


def test_get_coef_mat(mock_rover):
    # Check weight generation
    learner_ids, coef_mat = mock_rover._get_coef_mat()

    # Check that the aggregation is correct
    assert len(learner_ids) == len(coef_mat)

    for learner_id, coef_row in zip(learner_ids, coef_mat):
        # Offset of 1 since we have 1 fixed covariate from conftest
        # include the 0 fixed covariate
        indices = np.array(learner_id) + 1
        indices = np.hstack([0, indices])
        assert np.allclose(coef_row[indices], mock_rover.learners[learner_id].coef)
        assert np.allclose(np.delete(coef_row, indices), 0)


def test_get_super_coef(mock_rover):
    """Check the ensembled coefficients.

    Summary of performances, as defined by mock_rover:
    (1,3) - 1.2
    (2,3) - 1.0
    (3) - -0.3
    """

    single_model_coeffs = mock_rover._get_super_coef(
        max_num_models=10, kernel_param=10, ratio_cutoff=0.99
    )
    # With high cutoff, only single model is selected. Coefficients same as that single model
    best_learner_id = (1, 3)
    coef_index = mock_rover._get_coef_index(best_learner_id)
    expected_coef = np.zeros(5)
    expected_coef[coef_index] = mock_rover.learners[best_learner_id].coef

    assert np.allclose(single_model_coeffs, expected_coef)

    # Same with low max_num_models to consider
    lone_max_model_coeffs = mock_rover._get_super_coef(
        max_num_models=1, kernel_param=10, ratio_cutoff=0.99
    )
    assert np.allclose(lone_max_model_coeffs, single_model_coeffs)

    # Check that if all models considered, we'll be bound by the max/min of the individual
    # learners' coefficients
    all_models_means = mock_rover._get_super_coef(
        max_num_models=10, kernel_param=10, ratio_cutoff=-10
    )
    _, all_coeffs = mock_rover._get_coef_mat()
    assert np.all(all_models_means <= all_coeffs.max(axis=0))
    assert np.all(all_models_means >= all_coeffs.min(axis=0))
    # First covariate never represented in any sub learner, should have a 0 coefficient
    assert np.isclose(all_models_means[1], 0)


def test_get_super_learner(mock_rover):
    """Test that we can create a super learner object from rover after fitting."""
    mock_rover._get_learner = lambda learner_id, use_cache: Learner(
        model_class=model_type_dict["gaussian"],
        obs="y",
        param_specs={"mu": {"variables": list(range(5))}},
    )

    super_learner = mock_rover._get_super_learner(
        max_num_models=10, kernel_param=10, ratio_cutoff=-0.2
    )
    assert np.allclose(super_learner.coef, mock_rover._get_super_coef(10, 10, 0.2))
