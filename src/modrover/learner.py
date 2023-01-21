from __future__ import annotations

from operator import attrgetter
from typing import Callable, Optional
from warnings import warn

import numpy as np
from pandas import DataFrame
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable

from .globals import get_rmse, model_type_dict


class Learner:

    def __init__(
        self,
        learner_id: LearnerID,
        model_type: str,
        y: str,
        param_specs: dict[str, dict],
        offset: str = "offset",
        weights: str = "weights",
        model_eval_metric: Callable = get_rmse,
    ) -> None:
        """
        Initialize a Rover submodel

        :param learner_id: LearnerID, represents the covariate indices to fit on
        :param model_type: str, represents what type of model, e.g. gaussian, tobit, etc.
        :param col_obs: str, which column is the target column to predict out
        :param col_covs: list[str], all possible columns that rover can explore over
        :param col_offset:
        :param col_weights:
        :param model_param_name:
        :param model_eval_metric:
        :param optimizer_options:
        """
        self.learner_id = learner_id
        self.model_type = model_type

        # TODO: Should these be parameters to fit, or instance attributes?
        self.y = y
        # TODO: offset will be gone in the regmod v1.0.0
        self.offset = offset
        self.weights = weights
        self.model_eval_metric = model_eval_metric

        # Initialize null model
        self._model: Optional[RegmodModel] = None
        self.performance: Optional[float] = None

        # extract all covariates
        all_covariates = set()
        for param_spec in param_specs.values():
            all_covariates |= set(param_spec["variables"])
        self.all_covariates = list(all_covariates)

        # convert str to Variable
        # TODO: this won't be necessary in regmod v1.0.0
        for param_spec in param_specs.values():
            param_spec["variables"] = list(map(
                Variable, param_spec["variables"]
            ))
        self.param_specs = param_specs

    @property
    def cov_ids(self) -> tuple[int]:
        return self.learner_id.cov_ids

    @property
    def opt_coefs(self) -> Optional[np.ndarray]:
        if self._model:
            return self._model.opt_coefs
        return None

    @opt_coefs.setter
    def opt_coefs(self, opt_coefs: np.ndarray):
        if not self._model:
            self._model = self._initialize_model()
        self._model.opt_coefs = opt_coefs

    @property
    def model_class(self):
        if self.model_type not in model_type_dict:
            raise KeyError(f"Model type {self.model_type} not known, "
                           f"please select from {list(model_type_dict.keys())}")
        return model_type_dict[self.model_type]

    @property
    def vcov(self) -> Optional[np.ndarray]:
        if self._model:
            return self._model.opt_vcov
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

        # TODO: this shouldn't be necessary in regmod v1.0.0
        data = Data(
            col_obs=self.y,
            col_covs=self.all_covariates,
            col_offset=self.offset,
            col_weights=self.weights,
        )

        # Create regmod variables separately, by parameter
        # Initialize with fixed parameters
        # TODO: Does intercept get counted across multiple parameters?
        model = self.model_class(
            data=data,
            param_specs=self.param_specs,
        )
        self._model = model
        return model

    def fit(
        self,
        data: DataFrame,
        holdout_cols: Optional[list[str]] = None,
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

    def predict(self, model: RegmodModel, test_set: DataFrame) -> np.ndarray:
        """
        Wraps regmod's predict method to avoid modifying input dataset.

        Can be removed if regmod models use a functionally pure predict function, otherwise
        we will raise SettingWithCopyWarnings repeatedly.

        :param model: a fitted RegmodModel
        :param test_set: a dataset to generate predictions from
        :param param_name: a string representing the parameter we are predicting out on
        :return: an array of predictions for the model parameter of interest
        """
        df_pred = model.predict(test_set)
        col_pred = model.param_names[0]
        # TODO: in regmod v1.0.0, we should have a col called "pred_obs"
        # col_pred = "pred_obs"
        return df_pred[col_pred].to_numpy()

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

        preds = self.predict(model, test_set)
        observed = test_set[self.y]
        performance = self.model_eval_metric(
            obs=observed.array,
            pred=preds,
        )
        # Clear out attached dataframe from the model object
        model.data.detach_df()

        return performance

    def __repr__(self):
        return f"Learner({self.cov_ids})"
