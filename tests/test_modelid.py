from contextlib import nullcontext as does_not_raise

import pytest

from modrover.learnerid import LearnerID




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
def test_modelid_validation(input_cov_id, validated_cov_id, expectation):
    with expectation:
        modelid = LearnerID(input_cov_id)
        if validated_cov_id:
            assert modelid.cov_ids == validated_cov_id


def test_modelid_initialization():

    # Check that the correct cov_ids are set
    modelid = LearnerID((1, 2, 3))
    assert modelid.cov_ids == (0, 1, 2, 3)


def test_modelid_children():
    # Check children generation
    modelid = LearnerID((1, 2, 3))
    child_modelids = modelid.create_children(num_covs=5)
    assert len(child_modelids) == 2

    expected = [(0, 1, 2, 3, 4), (0, 1, 2, 3, 5)]
    assert all([child == expectation for child, expectation in zip(child_modelids, expected)])

    # Check no more children generated when all covariates are represented
    modelid2 = LearnerID((0, 1, 2, 3))
    assert not any(modelid2.create_children(num_covs=3))


def test_modelid_parents():
    # Check parent generation
    modelid = LearnerID((1, 2, 3))
    parents = modelid.create_parents()

    expected_parents = [(0, 2, 3), (0, 1, 3), (0, 1, 2)]
    assert len(parents) == 3
    assert all([parent == expectation for parent, expectation in zip(parents, expected_parents)])

    # assert that the base model has no parents
    modelid3 = LearnerID(())
    assert not any(modelid3.create_parents())
