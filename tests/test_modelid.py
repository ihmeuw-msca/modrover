from modrover.modelid import ModelID

import pytest


def test_modelid():
    # Validate out of bounds
    with pytest.raises(ValueError):
        ModelID((-1,))

    with pytest.raises(ValueError):
        ModelID((1, 2, 10), num_covs=5)

    # Check parent/children generation
    modelid = ModelID((1, 2, 3), num_covs=5)
    assert modelid.cov_ids == (0, 1, 2, 3)
    child_modelids = modelid.children
    assert len(child_modelids) == 2

    expected = [(0, 1, 2, 3, 4), (0, 1, 2, 3, 5)]
    assert all([child == expectation for child, expectation in zip(child_modelids, expected)])

    parents = modelid.parents

    expected_parents = [(0, 2, 3), (0, 1, 3), (0, 1, 2)]
    assert len(parents) == 3
    assert all([parent == expectation for parent, expectation in zip(parents, expected_parents)])

    # Check no more children generated when all covariates are represented
    modelid2 = ModelID((0, 1, 2, 3), num_covs=3)
    assert not any(modelid2.children)

    modelid3 = ModelID((), num_covs=3)
    assert not any(modelid3.parents)
