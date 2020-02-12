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


def bimodal(x, x01, sigma1, a1, x02, sigma2, a2):
    return gauss(x, a1, x01, sigma1) + gauss(x, a2, x02, sigma2)


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


def analysing_basin(dict):
    for key in dict:
        plt.figure()
        n, bins, patches = plt.hist(dict[key], rwidth=0.8)
        x = []
        for i in bins:
            x.append(i + 1)
        x = np.array(x[:-1])
        y = np.array(n[-1])

        n = len(dict[key])
        mean = np.mean(dict[key])
        sigma = np.std(dict[key])
        combined_sigma = sigma/np.sqrt(n)

        # plt.plot(x, gauss(x, a=max(y), x0=mean, sigma=combined_sigma), 'ro:',
        #          label=f"mean={round(mean, 2)}, sigmas={round(combined_sigma, 2)}")
        # plt.legend()

        plt.title(f"{key}, mean = {round(mean, 2)}, sigma = {round(sigma, 2)}")
        plt.xlabel("Glaciation Temperature (Degrees)")
        plt.ylabel("Frequency")


def analysing_x(array, title, fit="gauss"):
    plt.figure()
    n, bins, patches = plt.hist(array.values(), bins=np.arange(-44, -10, 2), rwidth=0.8,
                                histtype="barstacked", label=array.keys())
    plt.title(f"GT by {title}, n={sum(n[-1])}")
    x = []
    for i in bins:
       x.append(i + 1)
    x = np.array(x[:-1])
    y = np.array(n[-1])

    n = len(x)
    mean = np.mean(x)
    sigma = np.std(x)

    if fit == "gauss":
        popt, pcov = sp.curve_fit(gauss, x, y, p0=[1, mean, sigma])
        ss_res = np.sum((y - gauss(x,*popt))**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot
        plt.plot(x, gauss(x, *popt), 'ro:',
                 label=f"peaks={round(popt[1], 2)}, sigmas={round(popt[2], 2)}, r2={round(r2, 2)}")
    elif fit == "bimodal":
        popt, pcov = sp.curve_fit(bimodal, x, y, p0=[-37.5, 2.5, 8, -29, 2.5, 8])
        ss_res = np.sum((y - bimodal(x,*popt))**2)
        ss_tot= np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot
        testx = np.arange(min(x), max(x), 0.1)
        plt.plot(testx, bimodal(testx, *popt), ':',
                 label=f"peaks={round(popt[0], 2), round(popt[3], 2)}, sigmas={round(popt[1], 2), round(popt[4], 2)}, "
                       f"r2={round(r2, 2)}")

    plt.xlabel("Glaciation Temperature (Degrees)")
    plt.ylabel("Frequency")
    plt.legend()


if __name__ == "__main__":
    histogram_dict = []
    best_track_df = pd.read_csv(BEST_TRACK_CSV, skiprows=[1], na_values=" ", keep_default_na=False)
    for file in os.listdir("proc/eyes_since_2012"):
        filename = "proc/eyes_since_2012/" + file
        c = CycloneSnapshot.load(filename)
        try:
            c.mask_thin_cirrus()
            c.mask_array_I05(LOW=220, HIGH=270)
            c.mask_using_I01(80)
            gt, gt_err, r = c.gt_piece_percentile(percentile=5, plot=False)
            if gt is not np.nan and r > 0.85:
                c.unmask_array()
                max_I05 = c.I05.max()
                min_I05 = c.I05.max()
                cyc_date_minus_day = c.meta_data["ISO_TIME"] - timedelta(days=1)
                cyc_time_minus_day = cyc_date_minus_day.strftime("%Y-%m-%d %H:%M:%S")
                cyc_future = best_track_df.loc[(best_track_df["ISO_TIME"] > cyc_time_minus_day) &
                                               (best_track_df["NAME"] == c.meta_data["NAME"]) &
                                               (best_track_df["USA_SSHS"] > 1)]
                histogram_dict.append({"year": c.meta_data["SEASON"], "gt": gt, "r": r, "wind": c.meta_data["USA_WIND"],
                                       "cat": c.meta_data["USA_SSHS"], "basin": c.meta_data["BASIN"],
                                       "future_winds": cyc_future["USA_WIND"].values[:16],
                                       "position": [c.meta_data["LAT"], c.meta_data["LON"]],
                                       "time": c.meta_data["ISO_TIME"], "SST": max_I05, "CTT": min_I05})
        except:
            continue

    all_gts = []
    all_winds = []
    basin_gts = {"WP": [], "NA": [], "NI": [], "SI": [], "NP": [], "SA": [], "EP": [], "SP": []}
    increasing_basin_gts = {"WP": [], "NA": [], "NI": [], "SI": [], "NP": [], "SA": [], "EP": [], "SP": []}
    decreasing_basin_gts = {"WP": [], "NA": [], "NI": [], "SI": [], "NP": [], "SA": [], "EP": [], "SP": []}
    year_gts = {2012: [], 2013: [], 2014: [], 2015: [], 2016: [], 2017: [], 2018: [], 2019: []}
    cat_gts = {4.0: [], 5.0: []}
    wind_gts = {110: [], 120: [], 130: [], 140: [], 150: [], 160: []}
    max_wind_gts = {110: [], 120: [], 130: [], 140: [], 150: [], 160: []}
    WP_gts_by_lat = {9.5: [], 11.5: [], 13.5: [], 15.5: [], 17.5: [], 19.5: [],
                     21.5: [], 23.5: [], 25.5: [], 27.5: [], 29.5: [], 31.5: []}
    WP_gts_by_long = {110: [], 120: [], 130: [], 140: [], 150: []}
    season_gts = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [],  11: [], 12: []}
    increasing_gts = []
    decreasing_gts = []
    in_or_de_tensifying_gts = {"Increasing": [], "Decreasing" : [], "Max" : [], "Min": [], "Steady" :[]}
    for cyclone in histogram_dict:

        if cyclone["basin"] == "WP":
            if np.mean(cyclone["future_winds"][6:8]) < cyclone["wind"] < np.mean(cyclone["future_winds"][9:11]):
                in_or_de_tensifying_gts["Increasing"].append(cyclone["gt"])
            elif np.mean(cyclone["future_winds"][6:8]) > cyclone["wind"] > np.mean(cyclone["future_winds"][9:11]):
                in_or_de_tensifying_gts["Decreasing"].append(cyclone["gt"])
            elif np.mean(cyclone["future_winds"][6:8]) < cyclone["wind"] and np.mean(cyclone["future_winds"][9:11]) < cyclone["wind"]:
                in_or_de_tensifying_gts["Max"].append(cyclone["gt"])
            elif np.mean(cyclone["future_winds"][6:8]) > cyclone["wind"] and cyclone["wind"] < np.mean(cyclone["future_winds"][9:10]):
                in_or_de_tensifying_gts["Min"].append(cyclone["gt"])
            elif np.mean(cyclone["future_winds"][6:8]) == cyclone["wind"] and np.mean(cyclone["future_winds"][9:11]) == cyclone["wind"]:
                in_or_de_tensifying_gts["Steady"].append(cyclone["gt"])

            for lat in WP_gts_by_lat.keys():
                if lat - 1 < cyclone["position"][0] <= lat + 1:
                    WP_gts_by_lat[lat].append(cyclone["gt"])

            for long in WP_gts_by_long.keys():
                if long < cyclone["position"][1] <= long + 10:
                    WP_gts_by_long[long].append(cyclone["gt"])

        for wind in wind_gts.keys():
            if wind - 5 < cyclone["wind"] <= wind + 5:
                wind_gts[wind].append(cyclone["gt"])
            if wind + 10 > max(np.insert(cyclone["future_winds"], 0, cyclone["wind"])) >= wind:
                max_wind_gts[wind].append(cyclone["gt"])

        for month in season_gts.keys():
            if int(cyclone["time"].strftime("%m")) == month:
                season_gts[month].append(cyclone["gt"])

        all_gts.append(cyclone["gt"])
        all_winds.append(cyclone["wind"])

    analysing_x(in_or_de_tensifying_gts, "Intensity (WP)", fit="bimodal")
    # analysing_basin(season_gts)
    plt.show()
