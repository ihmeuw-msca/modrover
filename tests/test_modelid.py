from modrover.modelid import ModelID

import pytest


def test_modelid_validation():
    # Validate out of bounds
    with pytest.raises(ValueError):
        ModelID((-1,))

    with pytest.raises(ValueError):
        ModelID((1, 2, 10), num_covs=5)


def test_modelid_initialization():

    # Check that the correct cov_ids are set
    modelid = ModelID((1, 2, 3), num_covs=5)
    assert modelid.cov_ids == (0, 1, 2, 3)


def test_modelid_children():
    # Check children generation
    modelid = ModelID((1, 2, 3), num_covs=5)
    child_modelids = modelid.children
    assert len(child_modelids) == 2

    expected = [(0, 1, 2, 3, 4), (0, 1, 2, 3, 5)]
    assert all([child == expectation for child, expectation in zip(child_modelids, expected)])

    # Check no more children generated when all covariates are represented
    modelid2 = ModelID((0, 1, 2, 3), num_covs=3)
    assert not any(modelid2.children)


def test_modelid_parents():
    # Check parent generation
    modelid = ModelID((1, 2, 3), num_covs=5)
    parents = modelid.parents

    expected_parents = [(0, 2, 3), (0, 1, 3), (0, 1, 2)]
    assert len(parents) == 3
    assert all([parent == expectation for parent, expectation in zip(parents, expected_parents)])

    # assert that the base model has no parents
    modelid3 = ModelID((), num_covs=3)
    assert not any(modelid3.parents)
