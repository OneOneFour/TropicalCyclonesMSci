import pandas as pd
import os
from CycloneImage import get_eye

BEST_TRACK_CSV = os.environ.get("BEST_TRACK_CSV","data/best_fit_csv/ibtracs.last3years.list.v04r00.csv")

best_track_df = pd.read_csv(BEST_TRACK_CSV, skiprows=[1], na_values=" ", keep_default_na=False,
                            usecols=["SID", "ISO_TIME", "USA_SSHS", "LAT", "LON", "USA_STATUS", "USA_WIND", "USA_PRES",
                                     "USA_RMW", "BASIN", "NAME"])
best_track_df["ISO_TIME"] = pd.to_datetime(best_track_df["ISO_TIME"])
cat_4_5_all_basins = best_track_df.loc[
    (best_track_df["USA_SSHS"] > 3) & (best_track_df["ISO_TIME"] > pd.Timestamp(year=2017, month=9, day=17))]

cat_4_5_all_basins_group = cat_4_5_all_basins.groupby(["SID"])
print(len(cat_4_5_all_basins_group))
for name, cyclone in cat_4_5_all_basins_group:
    dict_cy = cyclone.to_dict(orient="records")
    for i, cyclone_point in enumerate(dict_cy[:-1]):
        start_point = cyclone_point
        end_point = dict_cy[i + 1]
        ci = get_eye(start_point, end_point, name=start_point["NAME"], basin=start_point["BASIN"],
                     cat=start_point["USA_SSHS"])
        if ci is not None:
            ci.draw_eye("I04")
            ci.save_object()
            ci.draw_rect((0,0),ci.rmw,ci.rmw)
    # for name_d, cyclone_day in cyclone:
    #     start, end = cyclone_day["ISO_TIME"].iloc[[0, -1]]
    #     margin = cyclone_day["USA_RMW"].mean() /30
    #     lat = cyclone_day["LAT"].mean()
    #     lon = cyclone_day["LON"].mean()
    #     ci = get_eye(start, end, (lat, lon),"N",margin)
    #     if ci is not None:
    #         ci.draw_eye("I04")
