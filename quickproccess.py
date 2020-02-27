import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

outdir = glob.glob("D:\out-240220\**\out.json", recursive=True)


def exp_dist(x, l, b):
    return l * np.exp(-(x + b) * l)


var = []
avg = []
exp_avg = []
var_avg = []
delta = []
eye = []
gt_grid = []
diff_eye_agg = np.array([])
diff_eye_median = []
diff_eye_mean = []
diff_eye_exp_mean = []
for file in outdir:
    with open(file) as f:
        obj = json.load(f)
        if obj["BASIN"] != "WP":
            continue
        if "GT_GRID" in obj and not np.isnan(obj["EYE"]):

            try:
                gt_i = np.array(obj["GT_GRID"])
                gt_i = gt_i[np.where(gt_i > -40)]

                pdf, edges, *others = np.histogram(gt_i, bins=100, normed=True)

                edges = (edges[1:] + edges[:-1]) / 2
                from scipy.optimize import curve_fit

                (l, b), coveff = curve_fit(exp_dist, edges, pdf)

                exp_avg.append(1 / l - b)
                var_avg.append((1 / l - b) ** 2)
            except RuntimeError:
                continue
            eye.append(obj["EYE"])
            avg.append(np.mean(obj["GT_GRID"]))
            var.append(np.var(obj["GT_GRID"]))

            diff_eye_agg = np.append(diff_eye_agg, np.array(obj["GT_GRID"]) - obj["EYE"])

            diff_eye_median.append(np.median(np.array(obj["GT_GRID"]) - obj["EYE"]))
            diff_eye_exp_mean.append(1/l -b - obj["EYE"])
            diff_eye_mean.append(np.mean(np.array(obj["GT_GRID"]) - obj["EYE"]))
            gt_grid.extend(obj["GT_GRID"])
        delta.append(obj["DELTA_SPEED_-24HR"])
        print(f"Loaded:{file}")
print(f"Loaded {len(outdir)} files")
# lb = np.array(lb)
# rb = np.array(rb)
# rf = np.array(rf)
# lf = np.array(lf)
eye = np.array(eye)
var = np.array(var)
avg = np.array(avg)
gt_grid = np.array(gt_grid)
diff_eye_mean = np.array(diff_eye_mean)
diff_eye_median = np.array(diff_eye_median)
diff_eye_exp_mean = np.array( diff_eye_exp_mean)
print(f"Average deviation from eye:{np.nanmean(eye - avg)}")
print(f"Average stderror : {np.nanstd(eye - avg) / np.sqrt(len(eye))}")


# print(f"Mean LF:{np.nanmean(lf)} std:{sem(lf)}")
# print(f"Mean RF:{np.nanmean(rf)} std:{sem(rf)}")
# print(f"Mean LB:{np.nanmean(lb)} std:{sem(lb)}")
# print(f"Mean  RB:{np.nanmean(rb)} std:{sem(rb)}")
#
# print(f"Avg std:{np.nanmean(var)}")


def plot_eye_distr():
    fig, ax = plt.subplots()
    ax.hist(eye, bins=6)
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
    ax.scatter(delta, diff_eye_median)
    ax.set_xlabel("Median cell difference to eye")
    ax.set_ylabel("12 hour Wind Speed change")


def plot_distr_of_mean(class_by_eye_gt=True):
    fig, ax = plt.subplots()
    if class_by_eye_gt:
        diff_eye_warm = diff_eye_mean[np.argwhere(eye > -30)]
        diff_eye_cold = diff_eye_mean[np.argwhere(eye <= -30)]
        bins = np.linspace(-10, 10, 20)
        ax.hist(diff_eye_warm, bins, label="Warm eyes")
        ax.hist(diff_eye_cold, bins, label="Cool eyes")
        plt.legend()
    else:
        ax.hist(diff_eye_mean)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Mean cell temperature minus eye temperature")


def plot_distr_of_median(class_by_eye_gt=True):
    fig, ax = plt.subplots()
    if class_by_eye_gt:
        diff_eye_warm = diff_eye_median[np.argwhere(eye > -30)]
        diff_eye_cold = diff_eye_median[np.argwhere(eye <= -30)]
        bins = np.linspace(-10, 10, 20)
        ax.hist(diff_eye_warm, bins, label="Warm eyes")
        ax.hist(diff_eye_cold, bins, label="Cool eyes")
        plt.legend()
    else:
        ax.hist(diff_eye_median)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Median cell temperature minus eye temperature")


def plot_windspeed_avg():
    fig, ax = plt.subplots()
    ax.scatter(diff_eye_exp_mean, delta)
    print(f"DIFF:{sps.pearsonr(diff_eye_mean,delta)}")
    ax.set_ylabel("12 hours change in wind speed (kts)")
    ax.set_xlabel("Mean difference in glaciation temperature (C)")


def plot_windspeed_eye():
    fig, ax = plt.subplots()
    ax.scatter(eye, delta)
    print(f"EYE:{sps.pearsonr(eye,delta)}")
    ax.set_ylabel("12 hours change in wind speed (kts)")
    ax.set_xlabel("Eye glaciation temperature (C)")


plot_external_distribution()
plot_distribution_of_temp_total(50)
plot_windspeed_avg()
plot_windspeed_eye()

