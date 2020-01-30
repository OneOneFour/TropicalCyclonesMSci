import os
from datetime import timedelta, datetime
from typing import List

import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar
import matplotlib.pyplot as plt
import os
import scipy.optimize as sp

from CycloneImage import get_eye, get_entire_cyclone, CycloneImage
from CycloneSnapshot import CycloneSnapshot

BEST_TRACK_CSV = os.environ.get("BEST_TRACK_CSV", "Data/ibtracs.since1980.list.v04r00.csv")
best_track_df = pd.read_csv(BEST_TRACK_CSV, skiprows=[1], na_values=" ", keep_default_na=False)
best_track_df["ISO_TIME"] = pd.to_datetime(best_track_df["ISO_TIME"])


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def all_cyclones_since(year, month, day, cat_min=4):
    cat_4_5_all_basins = best_track_df.loc[
        (best_track_df["USA_SSHS"] >= cat_min) & (
                best_track_df["ISO_TIME"] > pd.Timestamp(year=year, month=month, day=day))]
    cat_4_5_all_basins_group = cat_4_5_all_basins.groupby(["SID"])
    for name, cyclone in cat_4_5_all_basins_group:
        dict_cy = cyclone.to_dict(orient="records")
        for i, cyclone_point in enumerate(dict_cy[:-1]):
            start_point = cyclone_point
            end_point = dict_cy[i + 1]
            if end_point["ISO_TIME"] - start_point["ISO_TIME"] > timedelta(hours=3):
                continue
            with ProgressBar():
                ci = get_entire_cyclone(start_point, end_point)
                if ci:
                    ci.plot_globe()
                    return ci


def all_cyclone_eyes_since(year, month, day, cat_min=4):
    cat_4_5_all_basins = best_track_df.loc[
        (best_track_df["USA_SSHS"] >= cat_min) & (
                best_track_df["ISO_TIME"] > pd.Timestamp(year=year, month=month, day=day))]
    cat_4_5_all_basins_group = cat_4_5_all_basins.groupby(["SID"])
    for name, cyclone in cat_4_5_all_basins_group:
        dict_cy = cyclone.to_dict(orient="records")
        for i, cyclone_point in enumerate(dict_cy[:-1]):
            start_point = cyclone_point
            end_point = dict_cy[i + 1]
            if end_point["ISO_TIME"] - start_point["ISO_TIME"] > timedelta(hours=3):
                continue
            with ProgressBar():
                snapshot = get_eye(start_point, end_point)
                if snapshot:
                    snapshot.plot()
                    snapshot.save("test.snap")


def get_cyclone_eye_name_image(name, year, max_len=np.inf, pickle=False):
    df_cyclone = best_track_df.loc[(best_track_df["NAME"] == name) & (best_track_df["ISO_TIME"].dt.year == year)]
    dict_cy = df_cyclone.to_dict(orient="records")
    snap_list = []
    for i, cyclone_point in enumerate(dict_cy[:-1]):
        if len(snap_list) >= max_len:
            return snap_list
        start_point = cyclone_point
        end_point = dict_cy[i + 1]
        with ProgressBar():
            # ci = get_eye_cubic(start_point, end_point, name=NAME, basin=start_point["BASIN"],
            #                    cat=start_point["USA_SSHS"], dayOrNight="D")
            # if ci is not None:
            #     ci.draw_eye()
            #     return ci
            eye = get_eye(start_point, end_point)
            if eye:
                if pickle:
                    eye.save("proc/pickle_data/%s%i" % (name, len(snap_list)))
                snap_list.append(get_eye(start_point, end_point))
    return snap_list


def get_cyclone_by_name_date(name, start, end):
    df_cyclone = best_track_df.loc[
        (best_track_df["NAME"] == name) & (best_track_df["USA_SSHS"] > 3)
        & (best_track_df["ISO_TIME"] <= end) & (best_track_df["ISO_TIME"] >= start)
        ]
    dict_cy = df_cyclone.to_dict(orient="records")
    for i, cyclone_point in enumerate(dict_cy[:-1]):
        start_point = cyclone_point
        end_point = dict_cy[i + 1]
        cy = get_entire_cyclone(start_point, end_point)
        if cy:
            return cy


def get_cyclone_by_name(name, year, max_len=np.inf, pickle=False, shading=True) -> List[CycloneImage]:
    df_cyclone = best_track_df.loc[
        (best_track_df["NAME"] == name) & (best_track_df["ISO_TIME"].dt.year == year) & (best_track_df["USA_SSHS"] > 3)]
    dict_cy = df_cyclone.to_dict(orient="records")
    snap_list = []
    for i, cyclone_point in enumerate(dict_cy[:-1]):
        if len(snap_list) >= max_len:
            return snap_list
        start_point = cyclone_point
        end_point = dict_cy[i + 1]
        with ProgressBar():
            cy = get_entire_cyclone(start_point, end_point)
            if cy:
                eye = cy.draw_eye()
                eye.save("proc/pickle_data/")
                snap_list.append(cy)
                if cy.is_eyewall_shaded or not shading:
                    if pickle:
                        eye = cy.draw_eye()
                        eye.save("proc/pickle_data/%s%i" % (name, len(snap_list)))
                    snap_list.append(cy)
    return snap_list


if __name__ == "__main__":
    histogram_dict = []
    for file in os.listdir("proc/eyes_since_2012"):
        filename = "proc/eyes_since_2012/" + file
        c = CycloneSnapshot.load(filename)
        try:
            c.mask_thin_cirrus()
            c.mask_array_I05(LOW=220, HIGH=270)
            c.mask_using_I01(80)
            gt, r = c.gt_piece_percentile(percentile=5, plot=False)
            if gt is not np.nan and r > 0.90:
                histogram_dict.append({"year": c.meta_data["SEASON"], "gt": gt, "r": r, "wind": c.meta_data["USA_WIND"],
                                      "cat": c.meta_data["USA_SSHS"], "basin": c.meta_data["BASIN"]})
        except:
            continue

    all_gts = []
    all_winds = []
    basin_gts = {"WP": [], "NA": [], "NI": [], "SI": [], "NP": [], "SA": [], "EP": [], "SP": []}
    year_gts = {2012: [], 2013: [], 2014: [], 2015: [], 2016: [], 2017: [], 2018: [], 2019: []}
    cat_gts = {4.0: [], 5.0: []}
    wind_gts = {110: [], 120: [], 130: [], 140: [], 150: [], 160: []}
    for cyclone in histogram_dict:
        for basin in basin_gts.keys():
            if cyclone["basin"] == basin:
                basin_gts[basin].append(cyclone["gt"])

        for year in year_gts.keys():
            if int(cyclone["year"]) == year:
                year_gts[year].append(int(cyclone["gt"]))

        for cat in cat_gts.keys():
            if cyclone["cat"] == cat:
                cat_gts[cat].append(cyclone["gt"])

        for wind in wind_gts.keys():
            if wind - 5 < cyclone["wind"] <= wind + 5:
                wind_gts[wind].append(cyclone["gt"])

        all_gts.append(cyclone["gt"])
        all_winds.append(cyclone["wind"])

    n, bins, patches = plt.hist(basin_gts.values(), bins=np.arange(-44, -10, 2), rwidth=0.8, histtype="barstacked",
                                label=basin_gts.keys())

    x = []
    for i in bins:
        x.append(i + 1)
    x = np.array(x[:-1])
    y = np.array(n[-1])

    n = len(x)
    mean = np.mean(x)
    sigma = np.std(x)
    popt, pcov = sp.curve_fit(gauss, x, y, p0=[1, mean, sigma])
    ss_res = np.sum((y - gauss(x,*popt))**2)
    ss_tot= np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot
    plt.plot(x, gauss(x, *popt), 'ro:', label=f"mean={round(popt[1], 2)}, sigma={round(popt[2], 2)}, r2={round(r2, 2)}")
    plt.legend()
    plt.show()
