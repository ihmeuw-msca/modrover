from dataclasses import dataclass
from operator import attrgetter
from warnings import warn

import numpy as np
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable
from pandas import DataFrame
from typing import Callable, Dict, List, Optional, Tuple

from .globals import get_rmse
from .learnerid import LearnerID
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


class Learner:

    def __init__(
        self,
        model_id: LearnerID,
        model_type: str,
        col_obs: str,
        col_covs: List[str],
        col_fixed: Dict[str, List],
        # TODO: make col_offset a dictionary for 2 parameter model
        col_offset: str = "offset",
        col_weights: str = "weights",
        model_param_name: str = '',
        # RMSE a better default metric
        model_eval_metric: Callable = get_rmse,
    ) -> None:
        """
        Initialize a Rover submodel

        :param model_id: LearnerID, represents the covariate indices to fit on
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

        # TODO: Should these be parameters to fit, or instance attributes?
        self.col_obs = col_obs
        self.col_covs = col_covs
        self.col_fixed = col_fixed
        self.col_offset = col_offset
        self.col_weights = col_weights
        self.model_eval_metric = model_eval_metric

        # Default to first model parameter name if not specified
        if not model_param_name:
            model_param_name = self.model_class.param_names[0]
        self.model_param_name = model_param_name

        # Initialize null model
        self._model: Optional[RegmodModel] = None
        self.performance: Optional[float] = None

    @property
    def cov_ids(self) -> Tuple[int]:
        return self.model_id.cov_ids

    @property
    def opt_coefs(self) -> Optional[np.ndarray]:
        if self._model:
            return self._model.opt_coefs
        else:
            return None

    @opt_coefs.setter
    def opt_coefs(self, opt_coefs: np.ndarray):
        if not self._model:
            self._model = self._initialize_model()
        self._model.opt_coefs = opt_coefs

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
        Check if our fit method has been called, by checking whether the model is null.
        """
        return self.opt_coefs is not None

    @property
    def df_coefs(self) -> Optional[DataFrame]:
        if not self.opt_coefs:
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

    def _initialize_model(self) -> RegmodModel:
        """
        Initialize a regmod model based on the provided modelspecs.
        """

        # Validate that all parameters are represented
        if not set(self.col_fixed.keys()) == set(self.model_class.param_names):
            raise ValueError(
                f"Mismatch between specified parameter names {set(self.col_fixed.keys())} "
                f"and expected parameter names {set(self.model_class.param_names)}")

        # TODO: Column selection logic should live in the rover class, move out of here
        # Select the parameter-specific covariate columns
        all_covariates = {self.col_covs[i - 1] for i in self.cov_ids if i > 0}

        # Add on the fixed columns for every parameter
        for parameter, columns in self.col_fixed.items():
            for col in columns:
                all_covariates.add(col)

        data = Data(
            col_obs=self.col_obs,
            col_covs=list(all_covariates),
            col_offset=self.col_offset,
            col_weights=self.col_weights,
        )

        # Create regmod variables separately, by parameter
        # Initialize with fixed parameters
        # TODO: Does intercept get counted across multiple parameters?
        regmod_variables = {
            param_name: {
                'variables': [Variable(cov) for cov in covariates]
            }
            for param_name, covariates in self.col_fixed.items()
        }

        # Add the remaining columns specific to this model
        for cov_id in self.cov_ids:
            if cov_id > 0:
                regmod_variables[self.model_param_name]['variables'].append(
                    Variable(self.col_covs[cov_id - 1])
                )

        model = self.model_class(
            data=data,
            param_specs=regmod_variables
        )
        self._model = model
        return model

    def fit(
        self,
        data: DataFrame,
        holdout_cols: Optional[List[str]] = None,
        **optimizer_options
    ):
        """
        Fit a set of models on a series of holdout datasets.

        This method will fit a model over k folds of the dataset, where k is the length
        of the provided holdout_cols list. It is up to the user to decide the train-test
        splits for each holdout column.

        On each fold of the dataset, the trained model will predict out on the validation set
        and obtain a score. The averaged score across all folds becomes the model's overall
        performance.

        Finally, a model is trained with all data in order to generate the final coefficients.

        :param data: a dataframe containing the training data
        :param holdout_cols: which column names to iterate over for cross validation
        :return:
        """
        if self.performance:
            # Learner already has been fit, exit early
            return

        if holdout_cols:
            # If holdout cols are provided, loop through to calculate OOS performance
            performance = 0.
            for col in holdout_cols:
                model = self._initialize_model()
                # Subset the data
                train_set = data.loc[data[col] == 0]
                test_set = data.loc[data[col] == 1]
                self._fit(model, train_set, **optimizer_options)
                performance += self.score(test_set, model)
            performance /= len(holdout_cols)  # Divide by k to get an unweighted average

            # Learner performance is average performance across each k fold
            self.performance = performance

        # Fit final model with all data included
        self._model = self._initialize_model()
        self._fit(self._model, data, **optimizer_options)

        # If holdout cols not provided, use in sample score for the full data model
        if not holdout_cols:
            self.performance = self.score(data, self._model)

    def _fit(self, model: RegmodModel, data: DataFrame, **optimizer_options) -> None:

        model.attach_df(data)

        mat = model.mat[0]
        if np.linalg.matrix_rank(mat) < mat.shape[1]:
            warn(f"Singular design matrix {self.cov_ids=:}")
            return

        model.fit(**optimizer_options)

    def predict(self, model: RegmodModel, test_set: DataFrame, param_name: str) -> np.ndarray:
        """
        Wraps regmod's predict method to avoid modifying input dataset.

        Can be removed if regmod models use a functionally pure predict function, otherwise
        we will raise SettingWithCopyWarnings repeatedly.

        :param model: a fitted RegmodModel
        :param test_set: a dataset to generate predictions from
        :param param_name: a string representing the parameter we are predicting out on
        :return: an array of predictions for the model parameter of interest
        """
        split_coefficients = model.split_coefs(model.opt_coefs)
        # Select the index of the parameter of interest
        index = model.param_names.index(param_name)
        coefficients = split_coefficients[index]
        parameter = model.params[index]

        data = model.data
        data.attach_df(test_set)

        preds = parameter.get_param(coefficients, data)
        return preds

    def score(self, test_set: DataFrame, model: Optional[RegmodModel] = None) -> float:
        """
        Given a model and a test set, generate an aggregate score.

        Score is based on the provided evaluation metric, comparing the difference between
        observed and predicted values.

        :param test_set: The holdout test set to generate predictions from
        :param model: The fitted model to set predictions on
        :return: a score determined by the provided model evaluation metric
        """

        if model is None:
            model = self._model

        preds = self.predict(model, test_set, self.model_param_name)
        observed = test_set[self.col_obs]
        performance = self.model_eval_metric(
            obs=observed.array,
            pred=preds,
        )
        # Clear out attached dataframe from the model object
        model.data.detach_df()

        return performance
