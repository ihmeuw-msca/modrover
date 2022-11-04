from warnings import warn

import numpy as np
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable
from pandas import DataFrame
from typing import Optional, Tuple

from .info import ModelSpecs
from .modelid import ModelID


class Model:

    def __init__(self, model_id: ModelID, model_type, col_covs: Dict[str, List], col_fixed: Dict[str, List],
                 ):
        # unpack specs, pass in arguments manually
        self.model_id = model_id
        self.specs = specs
        self.performance: Optional[float] = None

        # Complication: Tobit model has 2 parameters, so multiple columns variable/multiple columns fixed,
        # different per parameter
        # For now, assume single parameter model only. Discuss how to extend later

        # Initialize the model
        # Use regmod 0.0.8; model data no longer needed
        self._model: Optional[RegmodModel] = model_type(*args, **kwargs)

    @property
    def cov_ids(self) -> Tuple[int]:
        return self.model_id.cov_ids

    @property
    def opt_coefs(self) -> Optional[np.ndarray]:
        if self._model:
            return self._model.opt_coefs
        else:
            return None

    @property
    def vcov(self) -> Optional[np.ndarray]:
        if self._model:
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

    @property
    def df_coefs(self) -> Optional[DataFrame]:
        if not self.has_been_fit:
            return None
        # Is this full structure necessary? Or just the means?
        data = DataFrame({
            "cov_name": map(attrgetter("name"), self._model.params[0].variables),
            "mean": self.opt_coefs,
            "sd": np.sqrt(np.diag(self.opt_vcov))
        })
        return data

    def _initialize_model(self, data: DataFrame) -> RegmodModel:
        """
        Initialize a regmod model based on the provided modelspecs.
        """
        if self._model:
            # Return early if we've already initialized the model
            return self._model
        col_covs = [self.specs.col_covs[i] for i in self.cov_ids]
        col_covs = [*self.specs.col_fixed_covs, *col_covs]
        data = Data(
            df=data,
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
        self._model = model
        return model

    def set_model_coefficients(self, opt_coefs: np.ndarray, vcov: np.ndarray) -> None:
        """
        If we already have a set of coefficients for a model, e.g. saved to disk,
        set the coefficients on the model to prevent unnecessary fit calls.
        """
        self._model.opt_coefs = opt_coefs
        self._model.opt_vcov = vcov

    def fit(self, data: DataFrame) -> None:
        if self.has_been_fit:
            return

        # We need to initialize the model with data, so delay initialization
        # until we want to fit with some data
        if not self._model:
            self._initialize_model(data)

        mat = self._model.mat[0]
        if np.linalg.matrix_rank(mat) < mat.shape[1]:
            warn(f"Singular design matrix {self.cov_ids=:}")
            return

        self._model.fit(**self.specs.optimizer_options)

    def predict(self, data: DataFrame) -> np.ndarray:
        if not self.has_been_fit:
            warn("The model has not been fit yet, so returning empty array.")
            return np.array([])

        return self._model.get_params(self.opt_coefs)[0]
