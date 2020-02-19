import glob
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

outdir = glob.glob("D:\out_new\**\out.json", recursive=True)

lb = []
rb = []
lf = []
rf = []
avg = []
var = []
delta = []
eye = []
gt_grid = []
diff_eye_agg = np.array([])
diff_eye_median = []
diff_eye_mean = []
for file in outdir:
    with open(file) as f:
        print(file)
        obj = json.load(f)
        l = []
        eye.append(obj["EYE"])
        if not np.isnan(obj["LB"]):
            lb.append(obj["LB"])
            l.append(obj["LB"])
        if not np.isnan(obj["RB"]):
            rb.append(obj["RB"])
            l.append(obj["RB"])
        if not np.isnan(obj["RF"]):
            rf.append(obj["RF"])
            l.append(obj["RF"])
        if not np.isnan(obj["LF"]):
            lf.append(obj["LF"])
            l.append(obj["LF"])
        if "GT_GRID" in obj and not np.isnan(obj["EYE"]):
            diff_eye_agg = np.append(diff_eye_agg, np.array(obj["GT_GRID"]) - obj["EYE"])
            diff_eye_median.append(np.median(np.array(obj["GT_GRID"]) - obj["EYE"]))
            diff_eye_mean.append(np.mean(np.array(obj["GT_GRID"]) - obj["EYE"]))
            gt_grid.extend(obj["GT_GRID"])
        delta.append(obj["DELTA_SPEED_12HR"])
        l = np.array(l)
        avg.append(l.mean())
        var.append(l.std())

lb = np.array(lb)
rb = np.array(rb)
rf = np.array(rf)
lf = np.array(lf)
eye = np.array(eye)
gt_grid = np.array(gt_grid)
diff_eye_mean = np.array(diff_eye_mean)
diff_eye_median = np.array(diff_eye_median)
print(f"Average deviation from eye:{np.nanmean(eye - avg)}")
print(f"Avergage stderror : {np.nanstd(eye - avg) / np.sqrt(len(eye))}")
print(f"Mean LF:{np.nanmean(lf)} std:{sem(lf)}")
print(f"Mean RF:{np.nanmean(rf)} std:{sem(rf)}")
print(f"Mean LB:{np.nanmean(lb)} std:{sem(lb)}")
print(f"Mean  RB:{np.nanmean(rb)} std:{sem(rb)}")

print(f"Avg std:{np.nanmean(var)}")


def plot_distribution_of_temp_total(bins=30):
    fig, ax = plt.subplots()
    ax.hist(diff_eye_agg, bins=bins)
    mean = np.mean(diff_eye_agg)
    median = np.median(diff_eye_median)
    ax.axvline(mean, c='r', label=f"Mean:{round(mean, 2)}pm{round(sem(diff_eye_agg), 2)}")
    ax.axvline(median, c='g', label=f"Median:{round(median, 2)}pm{round(1.253 * sem(diff_eye_agg), 2)}")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Cell temperature minus eye temperature")
    plt.legend()


def plot_external_distribution(bins=30):
    fig, ax = plt.subplots()
    ax.hist(gt_grid, bins=bins)
    mean = np.mean(gt_grid)
    median = np.median(gt_grid)
    ax.axvline(mean, c='r', label=f"Mean:{round(mean, 2)}pm{round(sem(gt_grid), 2)}")
    ax.axvline(median, c='g', label=f"Median:{round(median, 2)}pm{round(1.253 * sem(gt_grid), 2)}")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Cell temperature minus eye temperature")
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
    ax.scatter(delta, avg)
    ax.invert_yaxis()
    ax.set_xlabel("3 hours change in wind speed (kts)")
    ax.set_ylabel("External glaciation temperature")


def plot_windspeed_eye():
    fig, ax = plt.subplots()
    ax.scatter(delta, eye)
    ax.set_xlabel("3 hours change in wind speed (kts)")
    ax.set_ylabel("Eye glaciation temperature")


plot_distribution_of_temp_total(50)
plot_distr_of_mean()
plot_distr_of_median()
