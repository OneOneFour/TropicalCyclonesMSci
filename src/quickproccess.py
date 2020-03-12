import glob
import json
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

from GTFit import GTFit

outdir = glob.glob("/home/robert/TropicalCyclonesMSci/out-composites/**/**/out.json", recursive=True)


def exp_dist(x, l, b):
    return l * np.exp(-(x + b) * l)


df = pd.DataFrame(columns=["EYE", "GT_GRID_MEAN", "GT_GRID_VAR", "GT_GRID_EXP_MEAN", "GT_GRID_EXP_VAR",
                           "DIFF_EYE_MEDIAN", "DIFF_EYE_EXP_MEAN", "DIFF_EYE_MEAN",
                           "DELTA_SPEED_-24HR", "DELTA_SPEED_+24HR", "GT"])

diff_eye_agg = np.array([])
gt_grid = []
for file in outdir:
    with open(file) as f:
        obj = json.load(f)
        entry = obj
        if "GT_GRID" in obj and not np.isnan(obj["EYE"]):
            obj["GT_GRID"] = [a for a in obj["GT_GRID"] if 0 > a > -45]
            try:
                gt_i = np.array(obj["GT_GRID"])
                gt_i = gt_i[np.where(gt_i > -40)]

                pdf, edges, *others = np.histogram(gt_i, bins=100, normed=True)

                edges = (edges[1:] + edges[:-1]) / 2
                from scipy.optimize import curve_fit

                (l, b), coveff = curve_fit(exp_dist, edges, pdf)

                entry["GT_GRID_EXP_MEAN"] = 1 / l - b
                entry["GT_GRID_EXP_VAR"] = (1 / l - b) ** 2
            except RuntimeError:
                continue

            entry["GT_GRID_MEAN"] = np.mean(obj["GT_GRID"])
            entry["GT_GRID_VAR"] = np.var(obj["GT_GRID"])

            diff_eye_agg = np.append(diff_eye_agg, np.array(obj["GT_GRID"]) - obj["EYE"])

            entry["DIFF_EYE_MEDIAN"] = np.median(np.array(obj["GT_GRID"]) - obj["EYE"])
            entry["DIFF_EYE_EXP_MEAN"] = (1 / l - b - obj["EYE"])
            entry["DIFF_EYE_MEAN"] = (np.mean(np.array(obj["GT_GRID"]) - obj["EYE"]))

            gt_grid.extend(obj["GT_GRID"])

        entry["DELTA_SPEED_-24HR"] = obj["DELTA_SPEED_-24HR"] if "DELTA_SPEED_-24HR" in obj else np.nan
        entry["DELTA_SPEED_+24HR"] = obj["DELTA_SPEED_+24HR"] if "DELTA_SPEED_+24HR" in obj else np.nan
        df = df.append(entry, ignore_index=True)
        print(f"Loaded:{file}")

print(f"Loaded {len(df)} files")

print(f"Average deviation from eye:{np.nanmean(df['EYE'] - df['GT_GRID_MEAN'])}")
print(f"Average stderror : {np.nanstd(df['EYE'] - df['GT_GRID_MEAN']) / np.sqrt(len(df['EYE']))}")

ri_df = df[df["DELTA_SPEED_-24HR"] >= 30]
no_ri_df = df[df["DELTA_SPEED_-24HR"] < 30]


def bin_percentile(df, percentile):
    i5_raw = list(chain.from_iterable(df[f"EYE_{percentile}_PERCENTILE_I5"].tolist()))
    i4_raw = list(chain.from_iterable(df[f"EYE_{percentile}_PERCENTILE_I4"].tolist()))
    gt_fit = GTFit(i4_raw, i5_raw)
    return gt_fit.bin_data(np.mean, 1)


def plot_percentiles(fig, ax):
    warm_df = df[df["EYE"] > -32]
    cold_df = df[df["EYE"] <= -32]
    for p in (5, 50, 95):
        i5, i4 = bin_percentile(warm_df, p)
        ax.plot(i4, i5, label=f"warm at {p}th percentile")
        i5, i4 = bin_percentile(cold_df, p)
        ax.plot(i4, i5, label=f"cold at {p}th percentile")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.legend()


def plot_eye_distr():
    fig, ax = plt.subplots()
    ax.hist(df["EYE"], bins=6)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Eye Glaciation Temperature (C)")


def plot_distribution_of_temp_total(bins=30):
    fig, ax = plt.subplots()
    ax.hist(diff_eye_agg, bins=bins)
    mean = np.mean(diff_eye_agg)
    median = np.median(diff_eye_agg)
    ax.axvline(mean, c='r', label=f"Mean:{round(mean, 2)}pm{round(sps.sem(diff_eye_agg), 2)}")
    ax.axvline(median, c='g', label=f"Median:{round(median, 2)}pm{round(1.253 * sps.sem(diff_eye_agg), 2)}")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Cell temperature minus eye temperature")
    plt.legend()


def plot_external_distribution(bins=30):
    fig, ax = plt.subplots()
    ax.hist(gt_grid, bins=bins)
    mean = np.mean(gt_grid)
    median = np.median(gt_grid)
    ax.axvline(mean, c='r', label=f"Mean:{round(mean, 2)}±{round(sps.sem(gt_grid), 2)}°C")
    ax.axvline(median, c='g', label=f"Median:{round(median, 2)}±{round(1.253 * sps.sem(gt_grid), 2)}°C")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Cell glaciation temperature (C)")
    plt.legend()


def diff_against_ds():
    fig, ax = plt.subplots()
    ax.scatter(df["DELTA_SPEED_-24HR"], df["DIFF_EYE_MEDIAN"])
    ax.set_xlabel("Median cell difference to eye")
    ax.set_ylabel("Previous 24 hour wind speed change (kts)")


def gt_histogram(bins=20):
    fig, ax = plt.subplots()
    ax.hist(df["EYE"], bins=bins)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Glaciation Temperature (C)")


# def plot_distr_of_mean(class_by_eye_gt=True):
#     fig, ax = plt.subplots()
#     if class_by_eye_gt:
#         diff_eye_warm_df =
#         diff_eye_cold = df["DIFF_EYE_MEAN"][np.argwhere(df["EYE"] <= -30)]
#         bins = np.linspace(-10, 10, 20)
#         ax.hist(diff_eye_warm, bins, label="Warm eyes")
#         ax.hist(diff_eye_cold, bins, label="Cool eyes")
#         plt.legend()
#     else:
#         ax.hist(diff_eye_mean)
#     ax.set_ylabel("Frequency")
#     ax.set_xlabel("Mean cell temperature minus eye temperature")


# def plot_distr_of_median(class_by_eye_gt=True):
#     fig, ax = plt.subplots()
#     if class_by_eye_gt:
#         diff_eye_warm = diff_eye_median[np.argwhere(eye > -30)]
#         diff_eye_cold = diff_eye_median[np.argwhere(eye <= -30)]
#         bins = np.linspace(-10, 10, 20)
#         ax.hist(diff_eye_warm, bins, label="Warm eyes")
#         ax.hist(diff_eye_cold, bins, label="Cool eyes")
#         plt.legend()
#     else:
#         ax.hist(diff_eye_median)
#     ax.set_ylabel("Frequency")
#     ax.set_xlabel("Median cell temperature minus eye temperature")


def plot_windspeed_avg():
    fig, ax = plt.subplots()
    ax.scatter(df["DIFF_EYE_EXP_MEAN"], df["DELTA_SPEED_-24HR"])
    print(f"DIFF:{sps.pearsonr(df['DIFF_EYE_EXP_MEAN'], df['DELTA_SPEED_-24HR'])}")
    ax.set_ylabel("Past 24 hours change in wind speed (kts)")
    ax.set_xlabel("Mean difference in glaciation temperature (C)")


def plot_windspeed_eye():
    fig, ax = plt.subplots()
    ax.scatter(df["EYE"], df["DELTA_SPEED_-24HR"])
    print(f"EYE:{sps.pearsonr(df['EYE'], df['DELTA_SPEED_-24HR'])}")
    ax.set_ylabel("Past 24 hours change in wind speed (kts)")
    ax.set_xlabel("Eye glaciation temperature (C)")


def plot_windspeed_avg_ri():
    fig, ax = plt.subplots()
    ax.scatter(ri_df["DIFF_EYE_EXP_MEAN"], ri_df["DELTA_SPEED_-24HR"])
    print(f"DIFF:{sps.pearsonr(ri_df['DIFF_EYE_EXP_MEAN'], ri_df['DELTA_SPEED_-24HR'])}")
    ax.set_ylabel("Past 24 hours change in wind speed (kts)")
    ax.set_xlabel("Mean difference in glaciation temperature (C)")


def plot_windspeed_eye_ri():
    fig, ax = plt.subplots()
    ax.scatter(ri_df["EYE"], ri_df["DELTA_SPEED_-24HR"])
    print(f"EYE:{sps.pearsonr(ri_df['EYE'], ri_df['DELTA_SPEED_-24HR'])}")
    ax.set_ylabel("Past 24 hours change in wind speed (kts)")
    ax.set_xlabel("Eye glaciation temperature (C)")


fig, ax = plt.subplots()
plot_percentiles(fig, ax)
plt.show()
