from __future__ import annotations

from operator import attrgetter
from typing import Callable, Optional
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable

from .globals import get_rmse

LearnerID = tuple[int, ...]


class Learner:
    def __init__(
        self,
        model_type: type,
        obs: str,
        param_specs: dict[str, dict],
        offset: str = "offset",
        weights: str = "weights",
        model_eval_metric: Callable = get_rmse,
    ) -> None:
        """
        Initialize a Rover submodel

        :param model_type: str, represents what type of model, e.g. gaussian, tobit, etc.
        :param col_obs: str, which column is the target column to predict out
        :param col_covs: list[str], all possible columns that rover can explore over
        :param col_offset:
        :param col_weights:
        :param model_param_name:
        :param model_eval_metric:
        :param optimizer_options:
        """
        self.model_type = model_type

        # TODO: Should these be parameters to fit, or instance attributes?
        self.obs = obs
        # TODO: offset will be gone in the regmod v1.0.0
        self.offset = offset
        self.weights = weights
        self.model_eval_metric = model_eval_metric

        # Initialize null model
        self._model: Optional[RegmodModel] = None
        self.performance: Optional[float] = None

        # convert str to Variable
        # TODO: this won't be necessary in regmod v1.0.0
        for param_spec in param_specs.values():
            param_spec["variables"] = list(map(Variable, param_spec["variables"]))
        self.param_specs = param_specs

    @property
    def coef(self) -> Optional[NDArray]:
        if self._model:
            return self._model.opt_coefs
        return None

    @coef.setter
    def coef(self, coef: NDArray):
        if not self._model:
            self._model = self._initialize_model()
        self._model.opt_coefs = coef

    @property
    def vcov(self) -> Optional[NDArray]:
        if self._model:
            return self._model.opt_vcov
        return None

    @vcov.setter
    def vcov(self, vcov: NDArray):
        if not self._model:
            self._model = self._initialize_model()
        self._model.opt_vcov = vcov

    @property
    def has_been_fit(self) -> bool:
        """
        Check if our fit method has been called, by checking whether the model is null.
        """
        return self.coef is not None

    @property
    def df_coefs(self) -> Optional[DataFrame]:
        if not self.coef:
            return None
        # TODO: Update this datastructure to be flexible with multiple parameters.
        # Should reflect final all-data model, perhaps prefix with parameter name
        # Is this full structure necessary? Or just the means?
        data = DataFrame(
            {
                "cov_name": map(attrgetter("name"), self._model.params[0].variables),
                "mean": self.coef,
                "sd": np.sqrt(np.diag(self.vcov)),
            }
        )
        return data

    def _initialize_model(self) -> RegmodModel:
        """
        Initialize a regmod model based on the provided modelspecs.
        """

        # TODO: this shouldn't be necessary in regmod v1.0.0
        data = Data(
            col_obs=self.obs,
            col_offset=self.offset,
            col_weights=self.weights,
            subset_cols=False,
        )

        # Create regmod variables separately, by parameter
        # Initialize with fixed parameters
        model = self.model_type(
            data=data,
            param_specs=self.param_specs,
        )
        self._model = model
        return model

    def fit(
        self,
        data: DataFrame,
        holdouts: Optional[list[str]] = None,
        **optimizer_options,
    ):
        """
        Fit a set of models on a series of holdout datasets.

        This method will fit a model over k folds of the dataset, where k is the length
        of the provided holdouts list. It is up to the user to decide the train-test
        splits for each holdout column.

        On each fold of the dataset, the trained model will predict out on the validation set
        and obtain a score. The averaged score across all folds becomes the model's overall
        performance.

        Finally, a model is trained with all data in order to generate the final coefficients.

        :param data: a dataframe containing the training data
        :param holdouts: which column names to iterate over for cross validation
        :return:
        """
        if self.performance:
            # Learner already has been fit, exit early
            return
        if holdouts:
            # If holdout cols are provided, loop through to calculate OOS performance
            performance = 0.0
            for col in holdouts:
                model = self._initialize_model()
                # Subset the data
                train_set = data.loc[data[col] == 0]
                test_set = data.loc[data[col] == 1]
                self._fit(model, train_set, **optimizer_options)
                performance += self.score(test_set, model)
            performance /= len(holdouts)  # Divide by k to get an unweighted average

            # Learner performance is average performance across each k fold
            self.performance = performance

        # Fit final model with all data included
        self._model = self._initialize_model()
        self._fit(self._model, data, **optimizer_options)

        # If holdout cols not provided, use in sample score for the full data model
        if not holdouts:
            self.performance = self.score(data, self._model)

    def _fit(self, model: RegmodModel, data: DataFrame, **optimizer_options) -> None:
        model.attach_df(data)
        mat = model.mat[0]
        if np.linalg.matrix_rank(mat) < mat.shape[1]:
            warn(f"Singular design matrix")
            return

        model.fit(**optimizer_options)

    def predict(self, model: RegmodModel, test_set: DataFrame) -> NDArray:
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
        observed = test_set[self.obs]
        performance = self.model_eval_metric(
            obs=observed.array,
            pred=preds,
        )
        # Clear out attached dataframe from the model object
        model.data.detach_df()

        return performance
