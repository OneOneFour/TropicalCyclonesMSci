import os
from datetime import timedelta

import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar

from CycloneImage import get_eye, get_entire_cyclone

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


def get_cyclone_by_name(name, year, per_cyclone=None, max_len=np.inf, shading=False):
    df_cyclone = best_track_df.loc[
        (best_track_df["NAME"] == name) & (best_track_df["ISO_TIME"].dt.year == year) & (best_track_df["USA_SSHS"] > 3)]
    dict_cy = df_cyclone.to_dict(orient="records")
    vals_series = []
    for i, cyclone_point in enumerate(dict_cy[:-1]):
        if len(vals_series) >= max_len:
            break
        start_point = cyclone_point
        end_point = dict_cy[i + 1]
        # ci = get_eye_cubic(start_point, end_point, name=NAME, basin=start_point["BASIN"],
        #                    cat=start_point["USA_SSHS"], dayOrNight="D")
        # if ci is not None:
        #     ci.draw_eye()
        #     return ci
        try:
            cy = get_entire_cyclone(start_point, end_point)

            if cy:
                print(f"Cyclone:{cy.metadata['NAME']} on {cy.metadata['ISO_TIME']}")
                if not (shading and cy.is_eyewall_shaded):
                    vals = per_cyclone(cy)
                    vals_series.append(vals)
        except Exception:
            import traceback
            traceback.print_exc()
            continue

    return vals_series
