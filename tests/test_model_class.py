from modrover.info import ModelSpecs
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
    # Fill in other columns
    dataframe['intercept'] = 1
    dataframe['holdout'] = np.random.randint(0, 2, 25)
    dataframe = dataframe.reset_index()
    return dataframe


def test_model_fit(dataset, model_specs):
    # Arbitrary: select first 3 covariates out of 5
    model_id = ModelID(cov_ids=(0, 1, 2), num_covs=5)
    model = Model(model_id=model_id, specs=model_specs)
    # Check that model is "new"
    assert not model.has_been_fit
    assert model.opt_coefs is None

    # Fit the model, don't check for correctness
    model.fit(dataset)
    assert model.has_been_fit
    assert isinstance(model.opt_coefs, np.ndarray)
    assert isinstance(model.vcov, np.ndarray)

    preds = model.predict(dataset)
    assert isinstance(preds, np.ndarray)