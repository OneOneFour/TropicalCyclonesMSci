import pandas as pd
import numpy as np
from fetch_file import get_data

BEST_TRACK_CSV = "data/best track/ibtracs.last3years.list.v04r00.csv"

best_track_df = pd.read_csv(BEST_TRACK_CSV, skiprows=[1], na_values=" ", keep_default_na=False,
                            usecols=["SID", "ISO_TIME", "USA_SSHS", "LAT", "LON", "USA_STATUS", "USA_WIND", "USA_PRES",
                                     "USA_RMW", "BASIN", "NAME"])
best_track_df["ISO_TIME"] = pd.to_datetime(best_track_df["ISO_TIME"])
cat_4_5_all_basins = best_track_df.loc[
    (best_track_df["USA_SSHS"] > 3) & (best_track_df["ISO_TIME"] > pd.Timestamp(year=2017, month=9, day=17))]

cat_4_5_all_basins_group = cat_4_5_all_basins.groupby(["SID"])
print(len(cat_4_5_all_basins_group))
for name, cyclone in cat_4_5_all_basins_group:
    cyclone = cyclone.groupby(cyclone["ISO_TIME"].dt.date)
    for name_d, cyclone_day in cyclone:
        start, end = cyclone_day["ISO_TIME"].iloc[[0, -1]]
