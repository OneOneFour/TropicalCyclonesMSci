import os
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar
import matplotlib.pyplot as plt

from CycloneImage import get_eye, get_entire_cyclone, CycloneImage
from CycloneSnapshot import CycloneSnapshot

BEST_TRACK_CSV = os.environ.get("BEST_TRACK_CSV", "Data/ibtracs.last3years.list.v04r00.csv")
best_track_df = pd.read_csv(BEST_TRACK_CSV, skiprows=[1], na_values=" ", keep_default_na=False)
best_track_df["ISO_TIME"] = pd.to_datetime(best_track_df["ISO_TIME"])


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


def get_cyclone_eye_name_image(name, year, max_len=np.inf):
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
                eye.save("proc/pickle_data/%s%i" % (name, len(snap_list)))
                snap_list.append(get_eye(start_point, end_point))
    return snap_list


def get_cyclone_by_name(name, year, max_len=np.inf) -> List[CycloneImage]:
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
            # ci = get_eye_cubic(start_point, end_point, name=NAME, basin=start_point["BASIN"],
            #                    cat=start_point["USA_SSHS"], dayOrNight="D")
            # if ci is not None:
            #     ci.draw_eye()
            #     return ci
            cy = get_entire_cyclone(start_point, end_point)
            if cy:
                eye = cy.draw_eye()
                eye.save("proc/pickle_data/%s%i" % (name, len(snap_list)))
                snap_list.append(cy)
    return snap_list


if __name__ == "__main__":
    # cis = get_cyclone_by_name("IRMA", 2017, max_len=1)

    c = CycloneSnapshot.load("proc/pickle_data/IRMA0")

    c.mask_thin_cirrus()
    c.mask_array_I05()
    c.mask_half("left")
    c.mask_half("bottom")
    c.gt_fit()

    plt.show()

    # r = cis[0].draw_rectangle((16.5, -55.283), 250000, 250000)
    # r_2 = cis[0].draw_rectangle(((16.13, -61.9)), 100000, 250000)
