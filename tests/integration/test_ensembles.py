import numpy as np
import pytest

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


def test_covariate_matrix_generation(mock_rover):
    # Check weight generation
    learner_ids, coeff_means = mock_rover._generate_coefficients_matrix()

    # Check that the aggregation is correct
    assert len(learner_ids) == len(coeff_means)

    for learner_id, coeff_row in zip(learner_ids, coeff_means):
        assert np.allclose(
            coeff_row[list(learner_id)],
            mock_rover.learners[learner_id].opt_coefs
        )
        assert np.allclose(np.delete(coeff_row, learner_id),
                           np.zeros(mock_rover.num_covariates - len(learner_id)))


def test_generate_coefficient_means(mock_rover):
    """Check the ensembled coefficients.

    Summary of performances, as defined by mock_rover:
    (0,1,3) - 1.2
    (0,2,3) - 1.0
    (0, 3) - -0.3
    """

    single_model_coeffs = mock_rover._generate_ensemble_coefficients(
        max_num_models=10, kernel_param=10, ratio_cutoff=.99)
    # With high cutoff, only single model is selected. Coefficients same as that single model
    assert np.allclose(single_model_coeffs, np.array([.2, .4, 0, .6, 0]))

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
    # Last covariate never represented in any sub learner, should have a 0 coefficient
    assert np.isclose(all_models_means[-1], 0)


def test_superlearner_creation(mock_rover):
    """Test that we can create a super learner object from rover after fitting."""
    mock_rover.get_learner = lambda learner_id, use_cache: \
        Learner(learner_id=tuple(range(5)),
                model_type='gaussian',
                y='y',
                param_specs={'mu': {'variables': list(range(5))}})

    super_learner = mock_rover._create_super_learner(
        max_num_models=10,
        kernel_param=10,
        ratio_cutoff=-.2
    )
    assert np.allclose(super_learner.opt_coefs,
                       mock_rover._generate_ensemble_coefficients(10, 10, .2)
                       )
