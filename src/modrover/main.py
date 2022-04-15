from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pplkit.data.interface import DataInterface

from .info import RoverSpecs
from .modelhub import ModelHub
from .strategies import strategy_type_dict


class Rover:

    def __init__(self, specs: RoverSpecs, modelhub: ModelHub):
        self.specs = specs
        self.modelhub = modelhub
        self.strategies = [
            strategy_type_dict[name](self.modelhub)
            for name in self.specs.strategy_names
        ]

    def explore(self):
        for i, strategy in enumerate(self.strategies):
            strategy.implement(
                **self.specs.strategy_options.get(
                    self.specs.strategy_names[i], {}
                )
            )

    def collect(self) -> pd.DataFrame:
        sub_dirs = map(
            attrgetter("name"), self.modelhub.output_dir.iterdir()
        )
        return pd.concat([
            collect_result(
                self.modelhub.dataif.get_fpath(sub_dir, key="output"),
                required_covs=self.modelhub.specs.col_covs
            )
            for sub_dir in sub_dirs
        ], axis=0).reset_index(drop=True)

    def count(self) -> Dict:
        return{
            "num_models": len(list(self.modelhub.output_dir.iterdir())),
            "model_universe_size": 2**len(self.modelhub.specs.col_covs)
        }


def collect_result(
    output_dir: Path,
    required_covs: Optional[List[str]] = None
) -> pd.DataFrame:
    dataif = DataInterface(output=output_dir)
    df_coefs = dataif.load_output("coefs.csv")
    performance = dataif.load_output("performance.yaml")

    df_coefs = df_coefs[df_coefs["cov_name"].isin(required_covs)].copy()
    coefs = dict(zip(df_coefs["cov_name"], df_coefs["mean"]))
    coef_sds = dict(zip(df_coefs["cov_name"], df_coefs["sd"]))
    df_result = pd.DataFrame({
        "dir_name": output_dir.name,
        **{cov: coefs.get(cov, 0.0) for cov in required_covs},
        **{cov + "_sd": coef_sds.get(cov, 0.0) for cov in required_covs},
        **performance,
    }, index=[0])
    return df_result
