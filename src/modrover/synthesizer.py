from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .info import SynthSpecs


def metrics_to_weights(metrics: ArrayLike,
                       max_num_models: int,
                       kernel_param: float,
                       ratio_cutoff: float) -> ArrayLike:
    sort_indices = np.argsort(metrics)[::-1]
    indices = metrics >= metrics.max()*ratio_cutoff
    indices[sort_indices[max_num_models:]] = False
    weights = np.zeros(metrics.size)
    metrics = metrics[indices]
    metrics = 1 - metrics / metrics.max()
    metrics = np.exp(-kernel_param*metrics)
    weights[indices] = metrics / metrics.sum()
    return weights


def get_weighted_sd(means: ArrayLike, sds: ArrayLike, weights: ArrayLike):
    return np.sqrt(
        weights.dot(sds**2 + means**2) - (weights.dot(means))**2
    )


def synthesize(df: pd.DataFrame,
               synth_specs: SynthSpecs,
               required_covs: List[str],
               validate: bool = False) -> pd.DataFrame:
    for cov in list(synth_specs.cov_bounds.keys()):
        if cov not in required_covs:
            del synth_specs.cov_bounds[cov]
    if len(synth_specs.cov_bounds) > 0:
        valid = np.vstack([
            (df[cov] >= bounds[0]) & (df[cov] <= bounds[1])
            for cov, bounds in synth_specs.cov_bounds.items()
        ]).all(axis=0)
    if validate:
        df = df[valid].reset_index(drop=True)

    df["weights"] = metrics_to_weights(
        df[synth_specs.metric_type],
        synth_specs.max_num_models,
        synth_specs.kernel_param,
        synth_specs.ratio_cutoff
    )

    df_result = pd.DataFrame({
        "cov_name": required_covs,
        "mean": [df[cov].dot(df["weights"]) for cov in required_covs],
        "sd": [get_weighted_sd(df[cov], df[cov + "_sd"], df["weights"])
               for cov in required_covs],
        "num_present": [int((df[cov] != 0.0).sum()) for cov in required_covs],
        "num_valid": int(valid.sum()),
    })
    return df_result
