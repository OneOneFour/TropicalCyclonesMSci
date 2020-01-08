import os
from datetime import timedelta

import pandas as pd
from dask.diagnostics.progress import ProgressBar

from CycloneImage import get_eye, wrap

BEST_TRACK_CSV = os.environ.get("BEST_TRACK_CSV", "data/best_fit_csv/ibtracs.last3years.list.v04r00.csv")
best_track_df = pd.read_csv(BEST_TRACK_CSV, skiprows=[1], na_values=" ", keep_default_na=False,
                            usecols=["SID", "ISO_TIME", "USA_SSHS", "LAT", "LON", "USA_STATUS", "USA_WIND", "USA_PRES",
                                     "USA_RMW", "BASIN", "NAME", "STORM_SPEED", "STORM_DIR", "USA_LAT", "USA_LON"])
best_track_df["ISO_TIME"] = pd.to_datetime(best_track_df["ISO_TIME"])


def all_cyclones_since(year, month, day):
    cat_4_5_all_basins = best_track_df.loc[
        (best_track_df["USA_SSHS"] > 3) & (best_track_df["ISO_TIME"] > pd.Timestamp(year=year, month=month, day=day))]
    cat_4_5_all_basins["LON"] = cat_4_5_all_basins["LON"].map(wrap)
    cat_4_5_all_basins_group = cat_4_5_all_basins.groupby(["SID"])
    print(len(cat_4_5_all_basins_group))
    for name, cyclone in cat_4_5_all_basins_group:
        dict_cy = cyclone.to_dict(orient="records")
        for i, cyclone_point in enumerate(dict_cy[:-1]):
            start_point = cyclone_point
            end_point = dict_cy[i + 1]
            if end_point["ISO_TIME"] - start_point["ISO_TIME"] > timedelta(hours=3):
                continue
            with ProgressBar():
                ci = get_eye(start_point, end_point, name=start_point["NAME"], basin=start_point["BASIN"],
                             cat=start_point["USA_SSHS"], dayOrNight="D")
                if ci is not None:
                    # box is four times RMW
                    if ci.is_complete:
                        ci.draw_eye("I05")
                        ci.new_rect(f"da whole thing", (0, 0), ci.rmw * 2, ci.rmw * 2)
                        ci.draw_rect("da whole thing", plot=True)
                        ci.save_object()
                        # for y in range(-2,2):
                        #     for x in range(-2,2):
                        #         ci.draw_rect((ci.rmw/2 + ci.rmw*y, ci.rmw/2 + ci.rmw*x), ci.rmw, ci.rmw)


def cyclone_track(NAME):
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
                         cat=start_point["USA_SSHS"], dayOrNight="D",wind_speed =start_point["USA_WIND"])

            if ci is not None:
                ci.draw_eye()
                # return ci


all_cyclones_since(2012, 5, 1)
