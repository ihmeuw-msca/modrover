from itertools import combinations
from operator import attrgetter
from pathlib import Path
from typing import Dict, Optional
from warnings import warn

import numpy as np
from pandas import DataFrame
from pplkit.data.interface import DataInterface
from regmod.data import Data
from regmod.models import Model
from regmod.variable import Variable

from .info import ModelEval, ModelSpecs
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

    def _get_model(self,
                   cov_ids: CovIDs,
                   df_coefs: Optional[DataFrame] = None) -> Model:
        col_covs = [self.specs.col_covs[i - 1] for i in cov_ids]
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
        if df_coefs is not None:
            df_coefs = df_coefs.set_index("cov_name")
            model.opt_coefs = df_coefs.loc[col_covs, "mean"].to_numpy()
        return model

    def _fit_model(self,
                   cov_ids: CovIDs,
                   df: Optional[DataFrame] = None) -> DataFrame:
        sub_dir = self.get_sub_dir(cov_ids)
        if df is None:
            df = self.dataif.load_input(self.input_path.name)
            df = df[self.specs.col_holdout == 0].reset_index(drop=True)

        model = self._get_model(cov_ids)
        model.attach_df(df)
        mat = model.mat[0].to_numpy()
        if np.linalg.matrix_rank(mat) < mat.shape[1]:
            warn(f"Singular design matrix {cov_ids=:}")
            return
        model.fit(**self.specs.optimizer_options)

        df_coefs = DataFrame({
            "cov_name": map(attrgetter("name"), model.params[0].variables),
            "mean": model.opt_coefs,
            "sd": np.sqrt(np.diag(model.opt_vcov)),
        })

        self.dataif.dump_output(df_coefs, sub_dir, "coefs.csv")
        return df_coefs

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
        df_pred.rename(
            columns={self.specs.model_param_name: "pred"},
            inplace=True
        )
        col_kept = [
            *self.specs.col_id,
            self.specs.col_obs,
            "pred",
            # "Adjusted SMR, all deaths",
            # "Expected Deaths, all deaths",
        ]
        df_pred = df_pred[col_kept].copy()
        self.dataif.dump_output(df_pred, sub_dir, "result.parquet")
        return df_pred

    def _eval_model(self,
                    cov_ids: CovIDs,
                    df_pred: Optional[DataFrame] = None) -> Dict:
        sub_dir = self.get_sub_dir(cov_ids)
        if df_pred is None:
            df_pred = self.dataif.load_output(sub_dir, "result.parquet")
        obs = df_pred[self.specs.col_obs].to_numpy()
        pred = df_pred["pred"].to_numpy()
        # obs = df_pred["Adjusted SMR, all deaths"].to_numpy()
        # pred = (df_pred["pred"] / df_pred["Expected Deaths, all deaths"]).to_numpy()
        holdout = df_pred[self.specs.col_holdout].to_numpy()

        obs = self.eval.transformation(obs)
        pred = self.eval.transformation(pred)

        indices = {
            "insample": holdout == 0,
            "outsample": holdout == 1,
        }

        # pred = pred - pred[indices["insample"]].mean() + obs[indices["insample"]].mean()

        performance = {
            t: self.eval.metric(obs[indices[t]], pred[indices[t]])
            for t in ["insample", "outsample"]
        }
        self.dataif.dump_output(performance, sub_dir, "performance.yaml")
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

    def run_model(self, cov_ids: CovIDs):
        if self._has_ran(cov_ids):
            return
        df = self.dataif.load_input(self.input_path.name)
        df_train = df[df[self.specs.col_holdout] == 0].reset_index(drop=True)

        df_coefs = self._fit_model(cov_ids, df_train)
        if df_coefs is not None:
            df_pred = self._predict_model(cov_ids, df, df_coefs)
            self._eval_model(cov_ids, df_pred)

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
