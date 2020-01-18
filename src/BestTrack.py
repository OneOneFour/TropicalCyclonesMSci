from datetime import timedelta

import pandas as pd
import os
from CycloneImage import get_eye, wrap
from dask.diagnostics.progress import ProgressBar

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
                snapshot = get_eye(start_point, end_point)
                snapshot.save("")


def get_cyclone_name_images(NAME):
    df_cyclone = best_track_df.loc[(best_track_df["NAME"] == NAME) & (best_track_df["USA_SSHS"] > 3)]
    df_cyclone["LON"] = df_cyclone["LON"].map(wrap)
    dict_cy = df_cyclone.to_dict(orient="records")
    for i, cyclone_point in enumerate(dict_cy[:-1]):
        start_point = cyclone_point
        end_point = dict_cy[i + 1]
        with ProgressBar():
            # ci = get_eye_cubic(start_point, end_point, name=NAME, basin=start_point["BASIN"],
            #                    cat=start_point["USA_SSHS"], dayOrNight="D")
            # if ci is not None:
            #     ci.draw_eye()
            #     return ci
            ci = get_eye(start_point, end_point, name=NAME, basin=start_point["BASIN"],
                         cat=start_point["USA_SSHS"], dayOrNight="D", wind_speed=start_point["USA_WIND"])

            if ci is not None:
                ci.draw_eye()
                # return ci






if __name__ == "__main__":
    all_cyclones_since(2011, 1, 1)
