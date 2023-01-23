from contextlib import nullcontext as does_not_raise

import pytest

from modrover.learner import LearnerID


@pytest.mark.parametrize(
    "input_cov_id,validated_cov_id,expectation",
    [
        # Duplicated ints
        ((0, 1, 2, 3, 3, 3), (0, 1, 2, 3), does_not_raise()),
        # Include fixed if not present
        ((1, 2, 3), (0, 1, 2, 3), does_not_raise()),
        # Non ints
        (('1', '2', '3'), (0, 1, 2, 3), does_not_raise()),
        # Negatives should fail
        ((-1, 1, 3), None, pytest.raises(ValueError))
    ]
)
def test_learnerid_validation(input_cov_id, validated_cov_id, expectation):
    with expectation:
        learnerid = LearnerID(input_cov_id)
        if validated_cov_id:
            assert learnerid.cov_ids == validated_cov_id


def test_learnerid_initialization():

    # Check that the correct cov_ids are set
    learnerid = LearnerID((1, 2, 3))
    assert learnerid.cov_ids == (0, 1, 2, 3)


def test_learnerid_children():
    # Check children generation
    learnerid = LearnerID((1, 2, 3))
    child_learnerids = learnerid.create_children(num_covs=5)
    assert len(child_learnerids) == 2

    expected = [(0, 1, 2, 3, 4), (0, 1, 2, 3, 5)]
    assert all([child == expectation for child, expectation in zip(child_learnerids, expected)])

    # Check no more children generated when all covariates are represented
    learnerid2 = LearnerID((0, 1, 2, 3))
    assert not any(learnerid2.create_children(num_covs=3))


def test_learnerid_parents():
    # Check parent generation
    learnerid = LearnerID((1, 2, 3))
    parents = learnerid.create_parents()

    expected_parents = [(0, 2, 3), (0, 1, 3), (0, 1, 2)]
    assert len(parents) == 3
    assert all([parent == expectation for parent, expectation in zip(parents, expected_parents)])

    # assert that the base model has no parents
    learnerid3 = LearnerID(())
    assert not any(learnerid3.create_parents())
