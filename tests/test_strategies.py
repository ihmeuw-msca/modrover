from contextlib import nullcontext as does_not_raise

import pytest

from modrover.learner import Learner, LearnerID
from modrover.strategies import BackwardExplore, ForwardExplore, FullExplore
from modrover.strategies.base import RoverStrategy


class DummyModel(Learner):
    """Mock the Model class for testing. Only need a performance attribute."""

    def __init__(self, performance: float = 1.0):
        self.performance = performance


class DummyStrategy(RoverStrategy):

    def __init__(self, num_covs: int):
        self.num_covs = num_covs

    def generate_next_layer(self, *args, **kwargs):
        pass

    def get_upstream_learner_ids(self, *args, **kwargs):
        return set()


def test_basic_filtering():
    """Test that we can select the best learner IDs based on past performance."""
    num_covs = 5

    base_strategy = DummyStrategy(num_covs=num_covs)
    first_layer = [(0, i,) for i in range(1, num_covs + 1)]

    # Test 1: select the n best learners
    base_perf = 0
    delta = .2
    performances = {}
    for lid in first_layer:
        performances[lid] = DummyModel(base_perf)
        base_perf += delta

    best = base_strategy._filter_learner_ids(
        current_learner_ids=set(first_layer),
        learners=performances,
    )
    assert best == {first_layer[-1]}

    best_two = base_strategy._filter_learner_ids(
        current_learner_ids=set(first_layer),
        learners=performances,
        num_best=2
    )
    assert best_two == set(first_layer[-2:])


def test_parent_ratio():
    """Test that we can drop learner ids with better performing upstreams."""
    strategy = BackwardExplore(4)

    # Initialize a set of model ids and their children
    lid_1 = (0, 1)
    lid_2 = (0, 2)

    # Mock up some performances
    performances = {
        lid_1: DummyModel(5),
        lid_2: DummyModel(10)
    }

    upstreams = strategy.get_upstream_learner_ids(lid_1).union(
        strategy.get_upstream_learner_ids(lid_2)
    )
    for lid in upstreams:
        performances[lid] = DummyModel(7)

    # LID 1 should have worse performance than its upstreams, so should not be explored further
    # LID 2 has better performance than all parents, so we should be exploring further

    new_lids = strategy._filter_learner_ids(
        current_learner_ids={lid_1, lid_2},
        learners=performances,
        num_best=2,
        threshold=1
    )
    assert new_lids == {lid_2}


def test_generate_forward_layer():

    strategy = ForwardExplore(3)
    lid_1 = (0, 1)
    lid_2 = (0, 2)

    performances = {
        lid_1: DummyModel(),
        lid_2: DummyModel()
    }

    next_layer = strategy.generate_next_layer(
        current_learner_ids={lid_1, lid_2},
        learners=performances,
        num_best=2,
    )

    expected_layer = {
        (0, 1, 2),
        (0, 2, 3),
        (0, 1, 3)
    }
    assert next_layer == expected_layer

    # Check terminal condition
    terminal_lid = (0, 1, 2, 3)
    performances[terminal_lid] = DummyModel()
    final_layer = strategy.generate_next_layer(
        {terminal_lid},
        performances
    )
    assert not final_layer


def test_generate_backward_layer():
    strategy = BackwardExplore(3)
    lid_1 = (0, 1)
    lid_2 = (0, 2)

    performances = {
        lid_1: DummyModel(),
        lid_2: DummyModel()
    }

    next_layer = strategy.generate_next_layer(
        current_learner_ids={lid_1, lid_2},
        learners=performances,
        num_best=2,
    )

    expected_layer = {
        (0,)
    }
    assert next_layer == expected_layer

    # Check terminal condition
    terminal_lid = expected_layer.pop()
    performances[terminal_lid] = DummyModel()
    final_layer = strategy.generate_next_layer(
        {terminal_lid},
        performances
    )
    assert not final_layer


def test_full_explore():
    full_strategy = FullExplore(3)

    all_ids = set(full_strategy.generate_next_layer())
    expected_combos = {
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (0, 1, 2, 3),
    }
    expected_learner_ids = set(map(LearnerID, expected_combos))
    assert all_ids == expected_learner_ids

    # Check that a second call results in an empty generator
    second_layer = set(full_strategy.generate_next_layer())
    assert not second_layer


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
def test_as_learner_id(input_cov_id, validated_cov_id, expectation):
    strategy = FullExplore(3)
    with expectation:
        learner_id = strategy._as_learner_id(input_cov_id)
        if validated_cov_id:
            assert learner_id == validated_cov_id


def test_learnerid_initialization():
    strategy = FullExplore(3)
    # Check that the correct cov_ids are set
    learner_id = strategy._as_learner_id((1, 2, 3))
    assert learner_id == (0, 1, 2, 3)


def test_get_learner_id_children():
    strategy = FullExplore(5)
    # Check children generation
    learner_id = strategy._as_learner_id((1, 2, 3))
    children = strategy._get_learner_id_children(learner_id)
    assert len(children) == 2

    expected_children = {(0, 1, 2, 3, 4), (0, 1, 2, 3, 5)}
    assert expected_children == children

    # Check no more children generated when all covariates are represented
    strategy = FullExplore(3)
    learner_id = strategy._as_learner_id((0, 1, 2, 3))
    assert not any(strategy._get_learner_id_children(learner_id))


def test_learnerid_parents():
    strategy = FullExplore(5)
    # Check parent generation
    learner_id = strategy._as_learner_id((1, 2, 3))
    parents = strategy._get_learner_id_parents(learner_id)

    expected_parents = {(0, 2, 3), (0, 1, 3), (0, 1, 2)}
    assert len(parents) == 3
    assert expected_parents == parents

    # assert that the base model has no parents
    learner_id = strategy._as_learner_id(())
    assert not any(strategy._get_learner_id_parents(learner_id))
