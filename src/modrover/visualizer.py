from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize(df_coefs: pd.DataFrame,
              df_synth: pd.DataFrame,
              model_counts: Dict[str, int],
              required_covs: Dict[str, int]):
    np.random.seed(0)
    markers = {
        "init": "^",
        "final": "o",
    }

    # plot the coefficients
    fig, ax = plt.subplots(
        len(required_covs), 1, figsize=(8, 2*len(required_covs)), sharex=True
    )
    ax[0].set_title(
        (f"models="
         f"{model_counts['num_models']}/"
         f"{model_counts['model_universe_size']}"),
        loc="left"
    )
    metrics = df_coefs["outsample"].to_numpy()
    vmin, vmax = metrics.min(), metrics.max()
    indices = {
        "final": [
            len(dir_name.split("_")) == (len(required_covs) + 1)
            for dir_name in df_coefs["dir_name"]
        ]
    }
    df_synth = df_synth.set_index("cov_name")
    for i, cov in enumerate(required_covs):
        indices["init"] = [
            dir_name == f"0_{i + 1}"
            for dir_name in df_coefs["dir_name"]
        ]
        coefs = df_coefs[cov].to_numpy()
        ax[i].set_xlabel(cov)
        coefs_jitter = np.random.rand(coefs.size)
        im = ax[i].scatter(
            coefs, coefs_jitter,
            c=df_coefs["outsample"],
            alpha=0.2, vmin=vmin, vmax=vmax, edgecolor="none"
        )
        for key, value in markers.items():
            ax[i].scatter(
                coefs[indices[key]],
                coefs_jitter[indices[key]],
                marker=value, facecolor="none", edgecolor="grey",
                label=f"{key}_model"
            )
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='2%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.set_yticks([vmin, vmax])
        ax[i].plot(
            [df_synth.loc[cov, "mean"]]*2,
            [0, 1], color="#008080", linewidth=1,
            label=f"average={df_synth.loc[cov, 'mean']:.2f}"
        )
        ax[i].plot(
            [df_synth.loc[cov, "mean"] - 1.96*df_synth.loc[cov, "sd"],
             df_synth.loc[cov, "mean"] + 1.96*df_synth.loc[cov, "sd"]],
            [0.5]*2, color="#008080", linewidth=1,
            label=f"average_sd={df_synth.loc[cov, 'sd']:.2f}"
        )
        ax[i].axvline(0, linewidth=1, color="grey", linestyle="--")
        ax[i].legend(loc="upper left", bbox_to_anchor=(1.10, 1), fontsize=9)
        num_present = df_synth.loc[cov, "num_present"]
        num_valid = df_synth.loc[cov, "num_valid"]
        ax[i].text(
            0.02, 0.92,
            (f"present = {num_present}/{model_counts['num_models']}\n"
             f"valid = {num_valid}/{model_counts['num_models']}"),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax[i].transAxes,
            fontsize=9,
            bbox=dict(boxstyle='round',
                      facecolor="none", edgecolor="grey", alpha=0.5)
        )
