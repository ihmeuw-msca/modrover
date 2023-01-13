import numpy as np
import pandas as pd
import pytest

from modrover.learner import Learner
from modrover.learnerid import LearnerID


@pytest.fixture
def dataset():
    data = np.random.randn(25, 6)
    columns = [
        'var_a',
        'var_b',
        'var_c',
        'var_d',
        'var_e',
        'y']
    dataframe = pd.DataFrame(data, columns=columns)
    # Fill in intercept and holdout columns
    dataframe['intercept'] = 1
    dataframe['holdout_1'] = np.random.randint(0, 2, 25)
    dataframe['holdout_2'] = np.random.randint(0, 2, 25)
    return dataframe


@pytest.fixture
def model_specs():
    specs = dict(
        model_type='gaussian',
        y='y',
        param_specs={
            'mu': {"variables": ['intercept', 'var_a', 'var_b', 'var_c']}
        },
        offset='offset',
    )
    return specs


def test_model_init(model_specs):
    # Arbitrary: select first 2 covariates out of 5
    learner_id = LearnerID(cov_ids=(0, 1, 2, 3))
    model = Learner(learner_id=learner_id, **model_specs)
    # Check that model is "new"
    assert not model.has_been_fit
    assert model.opt_coefs is None
    assert model.performance is None

    regmod_model = model._initialize_model()

    # Should have 7 columns. y column, intercept, 3 covariates, weights, offset, trim
    assert regmod_model.data.df.shape == (0, 8)


def test_model_fit(dataset, model_specs):

    learner_id = LearnerID(cov_ids=(0, 1, 2, 3))
    model = Learner(learner_id=learner_id, **model_specs)

    # Fit the model, don't check for correctness
    model.fit(dataset, holdout_cols=['holdout_1', 'holdout_2'])
    assert 0 <= model.performance <= 1
    assert model.opt_coefs is not None
    assert isinstance(model.opt_coefs, np.ndarray)
    assert isinstance(model.vcov, np.ndarray)


def test_two_param_model_fit(dataset):

    # Sample two param model: a,b,c are mapped to mu, d,e to sigma

    learner_id = LearnerID(cov_ids=(0, 1, 2))

    model = Learner(
        learner_id=learner_id,
        model_type='tobit',
        y='y',
        param_specs={
            'mu': {
                "variables": ['intercept', 'var_a', 'var_b', 'var_c'],
            },
            'sigma': {
                "variables": ['intercept', 'var_d', 'var_e'],
            }
        },
    )

    # Should have 2 mu columns, 2 sigma columns, and the intercept
    regmod_model = model._initialize_model()
    assert set(regmod_model.data.col_covs) == {'var_a', 'var_b', 'var_c', 'var_d', 'var_e', 'intercept'}

    model.fit(dataset, holdout_cols=['holdout_1', 'holdout_2'])
    assert 0 <= model.performance <= 1
    assert model.opt_coefs is not None
    assert isinstance(model.opt_coefs, np.ndarray)
    assert isinstance(model.vcov, np.ndarray)


def test_initialize_model_with_coefs(model_specs):

    learner_id = LearnerID(cov_ids=(0, 1, 2, 3))
    model = Learner(learner_id=learner_id, **model_specs)

    # Set some known coefficients, random number
    # 3 covariates implies 3 coefficients
    expected_coefs = np.array([-.5, -.3, .3, .2])
    model.opt_coefs = expected_coefs
    assert np.isclose(model.opt_coefs, expected_coefs).all()

    with pytest.raises(ValueError):
        # Setting 4 coefficients on 3 variables should raise an error
        model.opt_coefs = np.append(expected_coefs, .4)
