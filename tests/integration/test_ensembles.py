import numpy as np
import pytest

from modrover.globals import model_type_dict
from modrover.learner import Learner
from modrover.synthesizer import metrics_to_weights


@pytest.mark.parametrize(
    "metrics,num_models,kernel_param,ratio_cutoff,expectation",
    [
        # Selecting a high cutoff means only 1 value selected
        (np.array([1, .5, 0]), 10, 1, .99, np.array([1, 0, 0])),
        # Two values should be selected with same max value, equal weights
        (np.array([1, .5, 1]), 10, 1, .99, np.array([.5, 0, .5])),
        # Lowering the threshold should return two values with unequal weights
        (np.array([1, .8, 0]), 10, 1, .7, np.array([0.55, 0.45, 0])),
        # Increasing the kernel parameter should bias towards the better performances
        (np.array([1, .8, 0]), 10, 10, .7, np.array([0.88, 0.12, 0])),
        # Lowering the num_models param should only return a max number of non zero weights
        (np.array([1, .8, 0]), 1, 1, .7, np.array([1, 0, 0])),
        # Negative values for performances handled gracefully
        (np.array([-.5, -.3, .2, .4]), 10, 1, -100, np.array([.06, .09, .32, .53]))
    ]
)
def test_create_weights(metrics, num_models, kernel_param, ratio_cutoff, expectation):
    weights = metrics_to_weights(
        metrics,
        num_models,
        kernel_param,
        ratio_cutoff
    )
    assert np.allclose(weights, expectation, atol=.01)
    assert np.isclose(weights.sum(), 1)


def test_learner_coefficient_mapping(mock_rover):
    assert mock_rover.num_covariates == 5

    # Given some faux coefficients, get the correct global coefficients
    # Since learner ids represent indices of explore columns, we have the expected mapping
    #   .1 = coefficient of col 0
    #   .3 = coefficient of col 1
    #   .2 = coefficient of col 4
    learner_id, coefficients = (0, 3), np.array([.1, .3, .2])

    row = mock_rover._learner_coefs_to_global_coefs(learner_id, coefficients)
    assert np.allclose(row, [.1, .3, 0, 0, .2])


def test_two_parameter_coefficient_mapping(mock_rover):
    """Test that covariate aggregation works for multiple parameter models."""

    mock_rover.model_type = "tobit"
    mock_rover.col_fixed = {
        'mu': [0, 3],
        'sigma': [4],
    }
    mock_rover.col_explore = [1, 2]
    mock_rover.explore_param = "mu"

    # Expected ordering out of a particular model:
    # Say learner_id = (0,) - represents both parameters
    # Expected covariate order is then [0, 3, 1, 4] - Tobit model defines mu -> sigma
    # as the parameter order
    # Rover appends the explore columns to the end of the mu fixed columns

    learner_id, coefficients = (0, ), np.array([.1, .3, .2, .4])

    row = mock_rover._learner_coefs_to_global_coefs(learner_id, coefficients)

    assert np.allclose(row, [.1, .3, .2, 0, .4])


def test_covariate_matrix_generation(mock_rover):
    # Check weight generation
    learner_ids, coeff_means = mock_rover._generate_coefficients_matrix()

    # Check that the aggregation is correct
    assert len(learner_ids) == len(coeff_means)

    for learner_id, coeff_row in zip(learner_ids, coeff_means):
        # Offset of 1 since we have 1 fixed covariate from conftest
        # include the 0 fixed covariate
        indices = np.array(learner_id) + 1
        indices = np.concatenate([np.zeros(1, dtype=np.int8), indices])
        assert np.allclose(
            coeff_row[indices],
            mock_rover.learners[learner_id].opt_coefs
        )
        assert np.allclose(np.delete(coeff_row, indices),
                           np.zeros(mock_rover.num_covariates - len(learner_id) - 1))


def test_generate_coefficient_means(mock_rover):
    """Check the ensembled coefficients.

    Summary of performances, as defined by mock_rover:
    (1,3) - 1.2
    (2,3) - 1.0
    (3) - -0.3
    """

    single_model_coeffs = mock_rover._generate_ensemble_coefficients(
        max_num_models=10, kernel_param=10, ratio_cutoff=.99)
    # With high cutoff, only single model is selected. Coefficients same as that single model
    best_learner_id = (1, 3)
    expected_coefs = mock_rover._learner_coefs_to_global_coefs(
        best_learner_id, mock_rover.learners[best_learner_id].opt_coefs
    )
    assert np.allclose(single_model_coeffs, expected_coefs)

    # Same with low max_num_models to consider
    lone_max_model_coeffs = mock_rover._generate_ensemble_coefficients(
        max_num_models=1, kernel_param=10, ratio_cutoff=.99)
    assert np.allclose(lone_max_model_coeffs, single_model_coeffs)

    # Check that if all models considered, we'll be bound by the max/min of the individual
    # learners' coefficients
    all_models_means = mock_rover._generate_ensemble_coefficients(
        max_num_models=10, kernel_param=10, ratio_cutoff=-10)
    _, all_coeffs = mock_rover._generate_coefficients_matrix()
    assert np.all(all_models_means <= all_coeffs.max(axis=0))
    assert np.all(all_models_means >= all_coeffs.min(axis=0))
    # First covariate never represented in any sub learner, should have a 0 coefficient
    assert np.isclose(all_models_means[1], 0)


def test_superlearner_creation(mock_rover):
    """Test that we can create a super learner object from rover after fitting."""
    mock_rover.get_learner = lambda learner_id, use_cache: \
        Learner(
            model_type=model_type_dict['gaussian'],
            y='y',
            all_covariates=list(range(5)),
            param_specs={'mu': {'variables': list(range(5))}}
        )

    super_learner = mock_rover._create_super_learner(
        max_num_models=10,
        kernel_param=10,
        ratio_cutoff=-.2
    )
    assert np.allclose(super_learner.opt_coefs,
                       mock_rover._generate_ensemble_coefficients(10, 10, .2)
                       )
