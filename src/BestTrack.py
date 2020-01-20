import os
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar

from CycloneImage import get_eye, get_entire_cyclone, CycloneImage

BEST_TRACK_CSV = os.environ.get("BEST_TRACK_CSV", "data/best_fit_csv/ibtracs.last3years.list.v04r00.csv")
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


def get_cyclone_eye_name_image(name, year):
    df_cyclone = best_track_df.loc[(best_track_df["NAME"] == name) & (best_track_df["ISO_TIME"].dt.year == year)]
    dict_cy = df_cyclone.to_dict(orient="records")
    snap_list = []
    for i, cyclone_point in enumerate(dict_cy[:-1]):
        start_point = cyclone_point
        end_point = dict_cy[i + 1]
        with ProgressBar():
            # ci = get_eye_cubic(start_point, end_point, name=NAME, basin=start_point["BASIN"],
            #                    cat=start_point["USA_SSHS"], dayOrNight="D")
            # if ci is not None:
            #     ci.draw_eye()
            #     return ci
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
                snap_list.append(cy)
    return snap_list

if __name__ == "__main__":
    cis, = get_cyclone_by_name("IRMA", 2017, max_len=1)

    r = cis.draw_rectangle((18.072, -55.572), 100000, 100000)
    r_2 = cis.draw_rectangle((15.8, -55.05), 200000, 200000)
    cis.plot_globe()