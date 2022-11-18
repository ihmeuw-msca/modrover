from modrover.model import Model
from modrover.modelid import ModelID

import numpy as np
import pandas as pd
import pytest

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
    # Fill in intercept
    dataframe['intercept'] = 1
    return dataframe


@pytest.fixture
def model_specs():
    specs = dict(
        model_type='gaussian',
        col_obs='y',
        col_fixed={'mu': ['intercept']},
        col_covs=['var_a', 'var_b', 'var_c', 'var_d', 'var_e'],
        model_param_name='mu'
    )
    return specs


def test_model_init(dataset, model_specs):
    # Arbitrary: select first 2 covariates out of 5
    model_id = ModelID(cov_ids=(0, 1, 2))
    model = Model(model_id=model_id, **model_specs)
    # Check that model is "new"
    assert not model.has_been_fit
    assert model.opt_coefs is None
    assert model.performance is None
    assert model.model_param_name == 'mu'

    # Only a and b are selected for this model
    assert set(model._model.data.col_covs) == \
           {'var_a', 'var_b', 'intercept'}

    # Should have 7 columns. y column, intercept, 2 covariates, weights, offset, trim
    assert model._model.data.df.shape == (0, 7)


def test_model_fit(dataset, model_specs):

    model_id = ModelID(cov_ids=(0, 1, 2))
    model = Model(model_id=model_id, **model_specs)

    # Fit the model, don't check for correctness
    model.fit(dataset)
    assert model.has_been_fit
    assert isinstance(model.opt_coefs, np.ndarray)
    assert isinstance(model.vcov, np.ndarray)

    preds = model.predict(dataset)
    assert isinstance(preds, pd.DataFrame)


def test_two_param_model_fit(dataset):

    # Sample two param model: a,b,c are mapped to mu, d,e to sigma

    model_id = ModelID(cov_ids=(0, 1, 2))

    model = Model(
        model_id=model_id,
        model_type='tobit',
        col_obs='y',
        col_covs=['var_a', 'var_b', 'var_c'],
        col_fixed={
            'mu': ['intercept'],
            'sigma': ['intercept', 'var_d', 'var_e']
        },
        model_param_name='mu',
    )

    # Should have 2 mu columns, 2 sigma columns, and the intercept
    assert set(model._model.data.col_covs) == {'var_a', 'var_b', 'var_d', 'var_e', 'intercept'}

    # TODO: check the data dimensions. If we want to fit a model on a/b/d/e,
    # should we see column C in the result?
    model.fit(dataset)

    preds = model.predict(dataset)
    assert isinstance(preds, pd.DataFrame)




