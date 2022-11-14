from operator import attrgetter
from warnings import warn

import numpy as np
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable
from pandas import DataFrame
from typing import Optional, Tuple

from .modelid import ModelID
from .info import model_type_dict



class Model:

    def __init__(
        self,
        model_id: ModelID,
        model_type: str,
        col_obs: str,
        col_covs: Tuple[str, ...],
        col_fixed_covs: Tuple[str, ...],
        col_holdout: Tuple[str, ...],
        col_offset: str = "offset",
        col_weights: str = "weights",
        col_eval_obs: str = "obs",
        col_eval_pred: str = "pred",
        optimizer_options: Optional[dict] = None,
    ) -> None:
        self.model_id = model_id
        self.model_type = model_type
        # For 2 parameter model: could be dict, or list of lists
        self.col_obs = col_obs
        self.col_covs = col_covs
        self.col_fixed_covs = col_fixed_covs
        self.col_holdout = col_holdout
        self.col_offset = col_offset
        self.col_weights = col_weights
        self.col_eval_obs = col_eval_obs
        self.col_eval_pred = col_eval_pred
        if optimizer_options is None:
            optimizer_options = {}
        self.optimizer_options = optimizer_options

        self.performance: Optional[float] = None

        # For now, assume single parameter model only. Discuss how to extend later
        # Look at pogit model and /mnt/team/msca/priv/jira/MSCA-205 notebooks
        # think about how to extend prediction/validation cases

        # Initialize the model
        self._model: RegmodModel = self._initialize_model()

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
    def model_class(self):
        try:
            return model_type_dict[self.model_type]
        except KeyError as e:
            raise KeyError(f"Model type {self.model_type} not known, "
                           f"please select from {list(model_type_dict.keys())}"
                           ) from e

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
            "sd": np.sqrt(np.diag(self.vcov))
        })
        return data

    def _initialize_model(self) -> RegmodModel:
        """
        Initialize a regmod model based on the provided modelspecs.
        """
        if self._model:
            # Return early if we've already initialized the model
            return self._model
        col_covs = [self.col_covs[i] for i in self.cov_ids]
        col_covs = [*self.col_fixed_covs, *col_covs]
        data = Data(
            col_obs=self.col_obs,
            col_covs=col_covs,
            col_offset=self.col_offset,
            col_weights=self.col_weights,
        )
        variables = [Variable(cov) for cov in col_covs]
        model = self.model_class(
            data,
            param_specs={
                self.model_param_name: {
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
        # Still unsure why vcov is necessary to set. From regmod code seems vcov is entirely
        # calculable from the opt_coefs. Plus, not settable anyways since vcov is a property
        # self._model.opt_vcov = vcov

    def fit(self, data: DataFrame) -> None:
        if self.has_been_fit:
            return

        # Only regmod 0.0.8+
        self._model.attach_df(data)

        mat = self._model.mat[0]
        if np.linalg.matrix_rank(mat) < mat.shape[1]:
            warn(f"Singular design matrix {self.cov_ids=:}")
            return

        self._model.fit(**self.optimizer_options)

    def predict(self, data: DataFrame) -> np.ndarray:
        if not self.has_been_fit:
            warn("The model has not been fit yet, so returning empty array.")
            return np.array([])

        return self._model.get_params(self.opt_coefs)[0]
