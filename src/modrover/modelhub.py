from itertools import combinations
from operator import attrgetter
from pathlib import Path
from typing import Dict, Optional
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame
from pplkit.data.interface import DataInterface
from regmod.data import Data
from regmod.models import Model
from regmod.variable import Variable

from .info import ModelEval, ModelSpecs
from .model import Model
from .modelid import ModelID
from .types import CovIDs, CovIDsSet


class ModelHub:

    def __init__(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        model_specs: ModelSpecs,
        model_eval: ModelEval
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.specs = model_specs
        self.eval = model_eval
        self.dataif = DataInterface(
            input=self.input_path.parent,
            output=self.output_dir
        )

    def _get_model(
            self, model_id: ModelID,
            previous_coefficients: Optional[np.array] = None
    ) -> Model:
        model = Model(model_id, self.specs)
        if previous_coefficients:
            model.set_model_coefficients(previous_coefficients)
        return model

    def _fit_model(self,
                   cov_ids: CovIDs,
                   df: Optional[DataFrame] = None) -> DataFrame:
        if df is None:
            df = self.dataif.load_input(self.input_path.name)
            df = df[self.specs.col_holdout == 0].reset_index(drop=True)

        model = self._get_model(cov_ids)
        model.fit(df)

    def _get_eval_obs(self, df: DataFrame) -> ArrayLike:
        return df[self.specs.col_obs]

    def _get_eval_pred(self, df: DataFrame) -> ArrayLike:
        return df[self.specs.model_param_name]

    def _predict_model(self,
                       cov_ids: CovIDs,
                       df: Optional[DataFrame] = None,
                       df_coefs: Optional[DataFrame] = None) -> DataFrame:
        sub_dir = self.get_sub_dir(cov_ids)
        if df is None:
            df = self.dataif.load_input(self.input_path.name)
        if df_coefs is None:
            df_coefs = self.dataif.load_output(sub_dir, "coefs.csv")
        model = self._get_model(cov_ids, df_coefs=df_coefs)

        df_pred = model.predict(df)
        df_pred[self.specs.col_eval_obs] = self._get_eval_obs(df_pred)
        df_pred[self.specs.col_eval_pred] = self._get_eval_pred(df_pred)
        df_pred = df_pred[self.specs.col_kept].copy()
        return df_pred

    def _eval_model(self,
                    cov_ids: CovIDs,
                    df_pred: Optional[DataFrame] = None,
                    col_holdout: Optional[str] = None) -> Dict:
        sub_dir = self.get_sub_dir(cov_ids)
        if df_pred is None:
            df_pred = self.dataif.load_output(sub_dir, "result.parquet")
        obs = df_pred[self.specs.col_eval_obs].to_numpy()
        pred = df_pred[self.specs.col_eval_pred].to_numpy()
        if col_holdout is not None:
            holdout = df_pred[col_holdout].to_numpy()
            indices = {
                "insample": holdout == 0,
                "outsample": holdout == 1,
            }
            performance = {
                t: self.eval.metric(obs[indices[t]], pred[indices[t]])
                for t in ["insample", "outsample"]
            }
        else:
            performance = {
                "insample": self.eval.metric(obs, pred)
            }
        return performance

    @staticmethod
    def get_sub_dir(cov_ids: CovIDs) -> str:
        return "_".join(map(str, ["0"] + sorted(cov_ids)))

    def _has_ran(self, cov_ids: CovIDs):
        sub_dir = self.get_sub_dir(cov_ids)
        outputs = [
            self.dataif.get_fpath(sub_dir, "coefs.csv", key="output"),
            self.dataif.get_fpath(sub_dir, "result.parquet", key="output"),
            self.dataif.get_fpath(sub_dir, "performance.yaml", key="output"),
        ]
        return all(f.exists() for f in outputs)

    def _run_model(self, cov_ids: CovIDs, col_holdout: Optional[str] = None):
        sub_dir = self.get_sub_dir(cov_ids)
        df = self.dataif.load_input(self.input_path.name)
        if col_holdout is not None:
            df_train = df[df[col_holdout] == 0].reset_index(drop=True)
            sub_dir = "/".join([sub_dir, col_holdout])
        else:
            df_train = df.copy()
        self._fit_model(cov_ids, df_train)
        df_coefs = model.df_coefs
        if df_coefs is not None:
            self.dataif.dump_output(df_coefs, sub_dir, "coefs.csv")
            df_pred = model.predict(cov_ids, df, df_coefs)
            self.dataif.dump_output(df_pred, sub_dir, "result.parquet")
            performance = self._eval_model(cov_ids, df_pred, col_holdout=col_holdout)
            self.dataif.dump_output(performance, sub_dir, "performance.yaml")

    def run_model(self, cov_ids: CovIDs):
        if self._has_ran(cov_ids):
            return
        sub_dir = self.get_sub_dir(cov_ids)
        # fit all the sub models for each holdout set
        for col_holdout in self.specs.col_holdout:
            self._run_model(cov_ids, col_holdout)
        # fit the full model
        self._run_model(cov_ids)
        # compute average out-of-sample score
        sub_dir_path = self.dataif.get_fpath(sub_dir, key="output")
        oos_dirs = [
            d.name for d in sub_dir_path.iterdir() if d.is_dir()
        ]
        performance = self.dataif.load_output(sub_dir, "performance.yaml")
        performance["outsample"] = sum([
            self.dataif.load_output(sub_dir, oos_dir, "performance.yaml")["outsample"]
            for oos_dir in oos_dirs
        ]) / len(oos_dirs)
        self.dataif.dump_output(performance, sub_dir, "performance.yaml")

    def get_model_performance(self,
                              cov_ids: CovIDs,
                              metric_type: str = "outsample") -> float:
        sub_dir = self.get_sub_dir(cov_ids)
        performance = self.dataif.load_output(sub_dir, "performance.yaml")
        return performance[metric_type]

    def get_child_cov_ids_set(
        self,
        cov_ids: CovIDs,
        cov_ids_set: Optional[CovIDsSet] = None
    ) -> CovIDsSet:
        rest_cov_ids = set(self.get_full_cov_ids()) - set(cov_ids)
        children = []
        for cov_id in rest_cov_ids:
            child = tuple(sorted([cov_id, *cov_ids]))
            if (cov_ids_set is None or child in cov_ids_set):
                children.append(child)
        return set(children)

    def get_parent_cov_ids_set(
        self,
        cov_ids: CovIDs,
        cov_ids_set: Optional[CovIDsSet] = None
    ) -> CovIDsSet:
        if len(cov_ids) == 0:
            return set()
        if len(cov_ids) == 1:
            return set([tuple()])
        parents = []
        for parent in combinations(cov_ids, len(cov_ids) - 1):
            parent = tuple(sorted(parent))
            if (cov_ids_set is None or parent in cov_ids_set):
                parents.append(parent)
        return set(parents)

    def get_full_cov_ids_set(self) -> CovIDsSet:
        cov_ids_set = []
        full_cov_ids = self.get_full_cov_ids()
        for i in range(len(self.specs.col_covs) + 1):
            cov_ids_set.extend(combinations(full_cov_ids, i))
        return set(cov_ids_set)

    def get_full_cov_ids(self) -> CovIDs:
        return tuple(range(1, len(self.specs.col_covs) + 1))

    def get_ran_cov_ids_set(self) -> CovIDsSet:
        cov_ids_set = []
        if self.dataif.output.exists():
            for sub_dir in self.dataif.output.iterdir():
                sub_dir = sub_dir.name
                cov_ids = tuple(map(int, sub_dir.split("_")))[1:]
                if self._has_ran(cov_ids):
                    cov_ids_set.append(cov_ids)
        return set(cov_ids_set)
