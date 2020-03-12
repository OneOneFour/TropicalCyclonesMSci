import ast
import atexit
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar

from CycloneImage import get_eye, get_entire_cyclone
from alt.CycloneMap import CycloneImageFast

if "NAUGHTY_LIST" in os.environ:
    try:
        with open(os.environ.get("NAUGHTY_LIST")) as naughty_file:
            NAUGHTY_LIST = ast.literal_eval(naughty_file.read())
            assert isinstance(NAUGHTY_LIST, set)
    except FileNotFoundError:
        NAUGHTY_LIST = set()


    @atexit.register
    def save_the_file():
        with open(os.environ.get("NAUGHTY_LIST"), 'w') as naughty_file:
            naughty_file.write(str(NAUGHTY_LIST))

BEST_TRACK_CSV = os.environ.get("BEST_TRACK_CSV",
                                r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\data\best_fit_csv\ibtracs.since1980.list.v04r00.csv")
best_track_df = pd.read_csv(BEST_TRACK_CSV, skiprows=[1], na_values=" ", keep_default_na=False)
best_track_df["ISO_TIME"] = pd.to_datetime(best_track_df["ISO_TIME"])
best_track_df = best_track_df[best_track_df.USA_RECORD != "L"]


def all_cyclones_since(year, month, day, cat_min=4, per_cyclone=None):
    cat_4_5_all_basins = best_track_df.loc[
        (best_track_df["USA_SSHS"] >= cat_min) & (
                best_track_df["ISO_TIME"] > pd.Timestamp(year=year, month=month, day=day))]
    cat_4_5_all_basins_group = cat_4_5_all_basins.groupby(["SID"])
    for sid, cyclone in cat_4_5_all_basins_group:
        dict_cy = cyclone.to_dict(orient="index")
        for index in list(dict_cy.keys())[:-1]:
            if index in NAUGHTY_LIST:
                continue
            start_point = dict_cy[index]
            if index + 1 not in dict_cy.keys():
                continue
            end_point = dict_cy[index + 1]
            history = best_track_df.loc[(best_track_df["SID"] == sid) &
                                        (best_track_df["ISO_TIME"] <= start_point["ISO_TIME"]) &
                                        (best_track_df["ISO_TIME"] > start_point["ISO_TIME"] - timedelta(
                                            hours=24))].to_dict(orient="records")
            future = best_track_df.loc[(best_track_df["SID"] == sid) &
                                       (best_track_df["ISO_TIME"] <= start_point["ISO_TIME"] + timedelta(hours=24)) & (
                                               best_track_df["ISO_TIME"] > start_point["ISO_TIME"])].to_dict(
                orient="records")
            try:
                ci = get_entire_cyclone(start_point, end_point, history=history, future=future)
                if ci and ci.is_eyewall_gt_good:
                    print(ci.metadata["NAME"])
                    per_cyclone(ci)
                else:
                    NAUGHTY_LIST.add(index)
            except Exception:
                import traceback
                traceback.print_exc()
                NAUGHTY_LIST.add(index)


# TODO: Move to using exclusion list
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


# # TODO: Move to using exclusion list
# def get_cyclone_eye_name_image(name, year, max_len=np.inf, pickle=False):
#     df_cyclone = best_track_df.loc[(best_track_df["NAME"] == name) & (best_track_df["ISO_TIME"].dt.year == year)]
#     dict_cy = df_cyclone.to_dict(orient="records")
#     snap_list = []
#     for i, cyclone_point in enumerate(dict_cy[:-1]):
#         if len(snap_list) >= max_len:
#             return snap_list
#         start_point = cyclone_point
#
#         end_point = dict_cy[i + 1]
#         with ProgressBar():
#             # ci = get_eye_cubic(start_point, end_point, name=NAME, basin=start_point["BASIN"],
#             #                    cat=start_point["USA_SSHS"], dayOrNight="D")
#             # if ci is not None:
#             #     ci.draw_eye()
#             #     return ci
#
#             eye = get_eye(start_point, end_point)
#             if eye:
#                 eye.mask_array_I05(275, 225)
#                 # eye.mask_thin_cirrus(80)
#
#                 if not np.isnan(eye.gt_piece_percentile(plot=False)).any():
#                     return eye
#     return snap_list


# TODO: Move to using exclusion list
def get_cyclone_by_name_date(name, start, end, per_cyclone=None):
    df_cyclone = best_track_df.loc[
        (best_track_df["NAME"] == name) & (best_track_df["USA_SSHS"] > 3)
        & (best_track_df["ISO_TIME"] <= end) & (best_track_df["ISO_TIME"] >= start)
        ]
    dict_cy = df_cyclone.to_dict(orient="index")
    for index in list(dict_cy.keys())[:-1]:
        start_point = dict_cy[index]
        if index + 1 not in dict_cy.keys():
            continue
        end_point = dict_cy[index + 1]
        try:
            cy = get_entire_cyclone(start_point,end_point,None,None)

        except FileNotFoundError:
            continue
        if cy:
            per_cyclone(cy)

def get_cyclone_by_name(name, year, per_cyclone=None, max_len=np.inf, shading=False):
    df_cyclone = best_track_df.loc[
        (best_track_df["NAME"] == name) & (best_track_df["ISO_TIME"].dt.year == year) & (best_track_df["USA_SSHS"] > 3)]
    dict_cy = df_cyclone.to_dict(orient="index")
    vals_series = []
    for index in list(dict_cy.keys())[:-1]:
        if len(vals_series) >= max_len:
            break
        if index in NAUGHTY_LIST:
            print(f"SKIPPING:{index}")
            continue
        start_point = dict_cy[index]
        if index + 1 not in dict_cy.keys():
            continue
        end_point = dict_cy[index + 1]
        history = best_track_df.loc[(best_track_df["NAME"] == name) &
                                    (best_track_df["ISO_TIME"] <= start_point["ISO_TIME"]) & (
                                            best_track_df["ISO_TIME"] > start_point["ISO_TIME"] - timedelta(
                                        hours=24))].to_dict(orient="records")
        future = best_track_df.loc[(best_track_df["NAME"] == name) &
                                   (best_track_df["ISO_TIME"] <= start_point["ISO_TIME"] + timedelta(hours=24)) & (
                                           best_track_df["ISO_TIME"] > start_point["ISO_TIME"])].to_dict(
            orient="records")
        # ci = get_eye_cubic(start_point, end_point, name=NAME, basin=start_point["BASIN"],
        #                    cat=start_point["USA_SSHS"], dayOrNight="D")
        # if ci is not None:
        #     ci.draw_eye()
        #     return ci
        try:
            cy = get_entire_cyclone(start_point, end_point, history=history, future=future)

            if cy and cy.is_eyewall_gt_good:
                print(f"Cyclone:{cy.metadata['NAME']} on {cy.metadata['ISO_TIME']}")
                if not (shading and cy.is_eyewall_shaded):
                    vals = per_cyclone(cy)
                    vals_series.append(vals)
            else:
                NAUGHTY_LIST.add(index)
        except Exception:
            import traceback
            traceback.print_exc()
            NAUGHTY_LIST.add(index)

    return vals_series
