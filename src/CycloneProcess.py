import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from GTFit import GTFit
from alt.CycloneMap import CycloneImageFast

CACHE_DIRECTORY = os.environ["CACHE_DIRECTORY"]


def compare_histograms():
    fig, axs = plt.subplots(1, 2)
    # Eye histogram
    axs[0].hist(eye_gts)
    axs[0].set_title("Eyewall $T_g$")
    axs[0].set_xlabel("Glaciation Temperature (C)")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(enviro_gts)
    axs[1].set_title("Environmental cell $T_g$")
    axs[1].set_xlabel("Glaciation Temperature (C)")
    axs[1].set_ylabel("Frequency")

    plt.show()


def plot_piecewise_composite():
    fig, ax = plt.subplots(figsize=(8, 8))

    gt_fit_eye = GTFit(eyes_i4, eyes_i5)
    gt_fit_eye.piecewise_percentile_multiple((5, 50, 95), plot_points=False, setup_axis=False, fig=fig, ax=ax,
                                             label="Eyewall")
    enviro_fit_eye = GTFit(environs_i4, environs_i5)
    enviro_fit_eye.piecewise_percentile_multiple((5, 50, 95), plot_points=False, fig=fig, ax=ax, setup_axis=False,
                                                 colors=["green", "orange", "blue"], label="External")
    ax.axhline(-38, c='y', label="$T_{g,homo}$", ls="--")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_ylabel("Cloud Temperature (C)")
    ax.set_xlabel("I4 band reflectance (K)")
    ax.legend()
    plt.show()


def overlap_BTD():
    fig, ax = plt.subplots()
    ax.scatter(eyes_i5, BTD_eyes, s=1, c="r", label="Eyewall")
    ax.scatter(environs_i5, BTD_enviro, s=0.05, c='b', label="External")
    ax.set_xlabel("I5 Temperature (C)")
    ax.set_ylabel("I4 - I5 (K)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    files = glob.glob(os.path.join(CACHE_DIRECTORY, "**.gzp"))
    percentiles = (5, 50, 95)
    eye_gts = []
    enviro_gts = []
    eyes_i4, eyes_i5 = [], []
    environs_i4, environs_i5 = [], []
    BTD_eyes, BTD_enviro = [], []
    # eyewall_BTD = []
    # eyes_i5 = []

    for f in files:
        try:
            ci = CycloneImageFast.from_gzp(f)
            # EYE
            if ci.eye().good_gt:
                # eye_i4, eye_i5 = ci.eye().bin_data_percentiles(percentiles)

                #
                # eyes_i4.extend(i4_f)
                eyes_i5.append(np.mean(ci.eye().i5_flat))
                BTD_eyes.append(np.mean(ci.eye().BTD))
                for c in ci.cells:
                    environs_i5.append(np.mean(c.i5_flat))
                    BTD_enviro.append(np.mean(c.BTD))

        except AssertionError:
            continue
    overlap_BTD()
