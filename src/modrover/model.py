from warnings import warn

import numpy as np
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable
from pandas import DataFrame
from typing import Optional

from .info import ModelSpecs
from .modelid import ModelID

class Model:

    def __init__(self, model_id: ModelID, specs: ModelSpecs):
        self.model_id = model_id
        self.specs = specs
        self.performance: Optional[float] = None

        # Initialize the model
        self._model: RegmodModel = self._initialize_model()

    @property
    def cov_ids(self) -> Tuple[int]:
        return self.model_id.cov_ids

    @property
    def opt_coefs(self) -> Optional[np.array]:
        return self._model.opt_coefs

    @property
    def vcov(self) -> Optional[np.array]:
        if self.has_been_fit:
            return self._model.opt_vcov
        else:
            return None

    @property
    def has_been_fit(self) -> bool:
        """
        Check if our model has already been fit by checking for the existence of
        coefficients.
        """
        return self.opt_coefs is not None

    def _initialize_model(self):
        """
        Initialize a regmod model based on the provided modelspecs.
        """
        col_covs = [self.specs.col_covs[i - 1] for i in self.cov_ids]
        col_covs = [*self.specs.col_fixed_covs, *col_covs]
        data = Data(
            col_obs=self.specs.col_obs,
            col_covs=col_covs,
            col_offset=self.specs.col_offset,
            col_weights=self.specs.col_weights,
        )
        variables = [Variable(cov) for cov in col_covs]
        model = self.specs.model_type(
            data,
            param_specs={
                self.specs.model_param_name: {
                    "variables": variables,
                    "use_offset": True,
                }
            }
        )
        return model

    def set_model_coefficients(self, opt_coefs: np.array) -> None:
        """
        If we already have a set of coefficients for a model, e.g. saved to disk,
        set the coefficients on the model to prevent unnecessary fit calls.
        """
        self._model.opt_coefs = opt_coefs

    def fit(self, data: DataFrame) -> None:
        if self.has_been_fit:
            return
        self._model.attach_df(data)
        mat = self._model.mat[0].to_numpy()
        if np.linalg.matrix_rank(mat) < mat.shape[1]:
            warn(f"Singular design matrix {self.cov_ids=:}")
            return

        self._model.fit(**self.specs.optimizer_options)

    def predict(self, data: DataFrame) -> DataFrame:
        df_pred = self._model.predict(data)
        return df_pred
