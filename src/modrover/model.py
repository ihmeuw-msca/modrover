from dataclasses import dataclass
from operator import attrgetter
from warnings import warn

import numpy as np
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable
from pandas import DataFrame
from typing import Callable, Dict, List, Optional, Tuple

from .globals import get_r2
from .modelid import ModelID
from .info import model_type_dict

@dataclass
class Rover:

    # Placeholder, to be fleshed out later and probably moved.
    model_type: str
    col_fixed: Dict[str, List]
    # col_fixed keyed by parameter, ex.
    # {
    #     'mu': ['intercept'],
    #     'sigma': []
    # }
    col_covs: List[str]  # All possible covariates we can explore over
    col_obs: str
    model_eval_metric: str


class Model:

    def __init__(
        self,
        model_id: ModelID,
        model_type: str,
        col_obs: str,
        col_covs: List[str],
        col_fixed: Dict[str, List],
        # TODO: make col_offset a dictionary for 2 parameter model
        col_offset: str = "offset",
        col_weights: str = "weights",
        model_param_name: str = '',
        model_eval_metric: Callable = get_r2,
        optimizer_options: Optional[dict] = None,
    ) -> None:
        """
        Initialize a Rover submodel

        :param model_id: ModelID, represents the covariate indices to fit on
        :param model_type: str, represents what type of model, e.g. gaussian, tobit, etc.
        :param col_obs: str, which column is the target column to predict out
        :param col_covs: List[str], all possible columns that rover can explore over
        :param col_fixed: Dict[str, List], which columns are invariant keyed by model parameter
        :param col_offset:
        :param col_weights:
        :param model_param_name:
        :param model_eval_metric:
        :param optimizer_options:
        """
        self.model_id = model_id
        self.model_type = model_type
        if optimizer_options is None:
            optimizer_options = {}
        self.optimizer_options = optimizer_options

        # Default to first model parameter name if not specified
        if not model_param_name:
            model_param_name = self.model_class.param_names[0]
        self.model_param_name = model_param_name

        # Initialize the model
        self._model: Optional[RegmodModel] = None
        self._initialize_model(
            col_obs=col_obs,
            col_covs=col_covs,
            col_fixed=col_fixed,
            col_offset=col_offset,
            col_weights=col_weights,
        )

        self.performance: Optional[float] = None

        # select appropriate evaluation callable
        self.model_eval_metric = model_eval_metric

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
        # TODO: Update this datastructure to be flexible with multiple parameters.
        # Should reflect final all-data model, perhaps prefix with parameter name
        # Is this full structure necessary? Or just the means?
        data = DataFrame({
            "cov_name": map(attrgetter("name"), self._model.params[0].variables),
            "mean": self.opt_coefs,
            "sd": np.sqrt(np.diag(self.vcov))
        })
        return data

    def _initialize_model(
        self,
        col_obs: str,
        col_covs: List[str],
        col_fixed: Dict[str, List],
        col_offset: str,
        col_weights,
    ) -> RegmodModel:
        """
        Initialize a regmod model based on the provided modelspecs.
        """
        if self._model:
            # Return early if we've already initialized the model
            return self._model

        # Validate that all parameters are represented
        if not set(col_fixed.keys()) == set(self.model_class.param_names):
            raise ValueError(
                f"Mismatch between specified parameter names {set(col_fixed.keys())} "
                f"and expected parameter names {set(self.model_class.param_names)}")

        # Select the parameter-specific covariate columns
        all_covariates = {col_covs[i - 1] for i in self.cov_ids if i > 0}

        # Add on the fixed columns for every parameter
        for parameter, columns in col_fixed.items():
            for col in columns:
                all_covariates.add(col)

        data = Data(
            col_obs=col_obs,
            col_covs=list(all_covariates),
            col_offset=col_offset,
            col_weights=col_weights,
        )

        # Create regmod variables separately, by parameter
        # Initialize with fixed parameters
        # TODO: Does intercept get counted across multiple parameters?
        regmod_variables = {
            param_name: {
                'variables': [Variable(cov) for cov in covariates]
            }
            for param_name, covariates in col_fixed.items()
        }

        # Add the remaining columns specific to this model
        for cov_id in self.cov_ids:
            if cov_id > 0:
                regmod_variables[self.model_param_name]['variables'].append(
                    Variable(col_covs[cov_id - 1])
                )

        model = self.model_class(
            data=data,
            param_specs=regmod_variables
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

    def fit(self, data):
        for col in self.col_holdout:
            model = self._initialize_model()
            self._fit(model, subset(data))
            self.score += self.score(model, subset(data))
        self.score /= len(self.col_holdout)
        model = self._initialize_model()
        self._fit(model, data)
        self.coefs = model.opt_coefs


    def _fit(self, data: DataFrame) -> None:
        if self.has_been_fit:
            return

        # Only regmod 0.0.8+
        for col in self.col_holdout:

            data = data.loc[data.col == 0]

            self._model.attach_df(data)

            mat = self._model.mat[0]
            if np.linalg.matrix_rank(mat) < mat.shape[1]:
                warn(f"Singular design matrix {self.cov_ids=:}")
                return

            self._model.fit(**self.optimizer_options)
            self.predict(self._model, data.loc[data.col == 1])
            self.update_score()


    def predict(self, data: DataFrame) -> np.ndarray:
        if not self.has_been_fit:
            raise RuntimeError("This model has not been fit yet, no coefficients "
                               "to predict from")

        return self._model.predict(data)

    def cross_validate(self):
        # Will need to think about how to do train/test splitting and what CV strategy to use.
        # standard k-fold validation? what value of k to use?
        # Any particular sampling strategy, just random?
        return NotImplemented

    def score(self, observed: np.ndarray, predicted: np.ndarray):
        # TODO: pass in observed/predicted, or pass in a whole validation set?
        if self.performance is None:
            # First time evaluating performance, score based on the predictions and the
            # scoring metric. If we've already scored just return the known number
            self.performance = self.model_eval_metric(
                obs=observed,
                predicted=predicted
            )

        return self.performance

# y    sdi   ldi   vac
#
#
# Rover(dataset, col_fixed={'mu': ['intercept'], 'sigma': ['intercept', 'vac']},
#       col_cov=['sdi', 'ldi'], parameter_to_explore='mu')
#
# Base Model -> Model
#
#
# Model._model: RegmodModel
#
# RegmodModel(
#
#     param_specs={
#         parameter_to_explore: [],
#         key: value for key, val in col_fixed
#     }
# )

