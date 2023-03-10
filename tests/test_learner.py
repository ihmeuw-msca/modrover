import numpy as np
import pandas as pd
import pytest

from modrover.globals import model_type_dict
from modrover.learner import Learner, LearnerID


@pytest.fixture
def dataset():
    data = np.random.randn(25, 6)
    covariate_columns = [
        'var_a',
        'var_b',
        'var_c',
        'var_d',
        'var_e',
        'y']
    dataframe = pd.DataFrame(data, columns=covariate_columns)
    # Fill in intercept and holdout columns
    dataframe['intercept'] = 1
    dataframe['holdout_1'] = np.random.randint(0, 2, 25)
    dataframe['holdout_2'] = np.random.randint(0, 2, 25)
    # Drop y covariate since it's the observed column
    return covariate_columns[:-1], dataframe


@pytest.fixture
def model_specs(dataset):
    all_covariates, _ = dataset
    specs = dict(
        model_type=model_type_dict['gaussian'],
        y='y',
        param_specs={
            'mu': {"variables": ['intercept', 'var_a', 'var_b', 'var_c']}
        },
        all_covariates=all_covariates[:-2],
        offset='offset',
    )
    return specs


def test_model_init(model_specs):
    # In param_specs, we'll select the intercept term plus 3/5 covariates
    model = Learner(**model_specs)
    # Check that model is "new"
    assert not model.has_been_fit
    assert model.opt_coefs is None
    assert model.performance is None

    regmod_model = model._initialize_model()

    # Should have 7 columns. y column, intercept, 3 covariates, weights, offset, trim
    assert regmod_model.data.df.shape == (0, 8)


def test_model_fit(dataset, model_specs):

    model = Learner(**model_specs)

    # Fit the model, don't check for correctness
    _, dataframe = dataset
    model.fit(dataframe, holdout_cols=['holdout_1', 'holdout_2'])
    assert 0 <= model.performance <= 1
    assert model.opt_coefs is not None
    assert isinstance(model.opt_coefs, np.ndarray)
    assert len(model.opt_coefs == 4)  # Intercept + 3 covariate terms
    assert isinstance(model.vcov, np.ndarray)


def test_two_param_model_fit(dataset):

    # Sample two param model: a,b,c are mapped to mu, d,e to sigma
    covariate_cols, dataframe = dataset
    model = Learner(
        model_type=model_type_dict['tobit'],
        y='y',
        param_specs={
            'mu': {
                "variables": ['intercept', 'var_a', 'var_b', 'var_c'],
            },
            'sigma': {
                "variables": ['intercept', 'var_d', 'var_e'],
            }
        },
        all_covariates=covariate_cols,
    )

    # Should have 2 mu columns, 2 sigma columns, and the intercept
    regmod_model = model._initialize_model()
    assert set(regmod_model.data.col_covs) == {'var_a', 'var_b', 'var_c', 'var_d', 'var_e', 'intercept'}

    model.fit(dataframe, holdout_cols=['holdout_1', 'holdout_2'])
    assert 0 <= model.performance <= 1
    assert model.opt_coefs is not None
    assert isinstance(model.opt_coefs, np.ndarray)
    # Result: intercept + 5 covariates + intercept = 7 coefficients
    assert len(model.opt_coefs) == 7
    assert isinstance(model.vcov, np.ndarray)


def test_initialize_model_with_coefs(model_specs):

    model = Learner(**model_specs)

    # Set some known coefficients, random number
    # Intercept + 3 covariates implies 4 coefficients
    expected_coefs = np.array([-.5, -.3, .3, .2])
    model.opt_coefs = expected_coefs
    assert np.isclose(model.opt_coefs, expected_coefs).all()

    with pytest.raises(ValueError):
        # Setting 5 coefficients on 4 variables should raise an error
        model.opt_coefs = np.append(expected_coefs, .4)
