import numpy as np
import pytest

from modrover.globals import model_type_dict
from modrover.learner import Learner


@pytest.mark.parametrize(
    "scores,top_pct_scores,top_pct_learners,expectation",
    [
        # Selecting a high cutoff means only 1 value selected
        (np.array([1, 0.5, 0]), 0.1, 1, np.array([1, 0, 0])),
        # Two values should be selected with same max value, equal weights
        (np.array([1, 0.5, 1]), 0.1, 1, np.array([0.5, 0, 0.5])),
        # Lowering the threshold should return two values with unequal weights
        (np.array([1, 0.5, 0]), 0.5, 1, np.array([0.67, 0.33, 0])),
        # Increasing the kernel parameter should bias towards the better scores
        (np.array([1, 0.5, 0]), 1, 0.1, np.array([1, 0, 0])),
        # Lowering the num_models param should only return a max number of non zero weights
        (np.array([1, 0.5, 0]), 1, 0.7, np.array([0.67, 0.33, 0])),
    ],
)
def test_create_weights(
    mock_rover, scores, top_pct_scores, top_pct_learners, expectation
):
    learner_ids = list(mock_rover.learners.keys())
    for learner_id, score in zip(learner_ids, scores):
        mock_rover.learners[learner_id].score = score
    weights = mock_rover._get_super_weights(
        learner_ids, top_pct_scores, top_pct_learners
    )
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


def test_get_super_coef(mock_rover):
    """Check the ensembled coefficients.

    Summary of scores, as defined by mock_rover:
    (1,3) - 1.2
    (2,3) - 1.0
    (3) - -0.3
    """

    learner_ids = list(mock_rover.learners.keys())
    weights = mock_rover._get_super_weights(
        learner_ids, top_pct_score=0.01, top_pct_learner=1
    )
    single_learner_coef = mock_rover._get_super_coef(learner_ids, weights)
    # With high cutoff, only single model is selected. Coefficients same as that single model
    best_learner_id = (1, 3)
    coef_index = mock_rover._get_coef_index(best_learner_id)
    expected_coef = np.zeros(5)
    expected_coef[coef_index] = mock_rover.learners[best_learner_id].coef

    assert np.allclose(single_learner_coef, expected_coef)

    # Same with low max_num_models to consider
    weights = mock_rover._get_super_weights(
        learner_ids, top_pct_score=1, top_pct_learner=0.01
    )
    lone_max_learner_coef = mock_rover._get_super_coef(learner_ids, weights)
    assert np.allclose(lone_max_learner_coef, single_learner_coef)

    # Check that if all models considered, we'll be bound by the max/min of the individual
    # learners' coefficients
    weights = mock_rover._get_super_weights(
        learner_ids, top_pct_score=1, top_pct_learner=1
    )
    all_learners_avg_coef = mock_rover._get_super_coef(learner_ids, weights)
    # First covariate never represented in any sub learner, should have a 0 coefficient
    assert np.isclose(all_learners_avg_coef[1], 0)


def test_get_super_learner(mock_rover):
    """Test that we can create a super learner object from rover after fitting."""
    mock_rover._get_learner = lambda learner_id, use_cache: Learner(
        model_class=model_type_dict["gaussian"],
        obs="y",
        main_param="mu",
        param_specs={"mu": {"variables": list(range(5))}},
    )

    learner_ids = list(mock_rover.learners.keys())
    weights = mock_rover._get_super_weights(learner_ids, 1, 1)
    super_learner = mock_rover._get_super_learner(
        top_pct_score=1, top_pct_learner=1, coef_bounds=None
    )
    assert np.allclose(
        super_learner.coef, mock_rover._get_super_coef(learner_ids, weights)
    )
    assert np.allclose(
        super_learner.vcov,
        mock_rover._get_super_vcov(learner_ids, weights, super_learner.coef),
    )
