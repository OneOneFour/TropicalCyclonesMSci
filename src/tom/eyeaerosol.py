import glob
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from AerosolImage import AerosolImageMODIS
from BestTrack import best_track_df
from CycloneSnapshot import CycloneSnapshot
plt.ioff()
files = glob.glob("C:\\Users\\Robert\\PycharmProjects\\TropicalCyclonesMSci\\tom_pickles\\**")
processed_df = pd.DataFrame()
print(len(files))
for file in files:
    cs = CycloneSnapshot.load(file)
    cs.metadata = cs.meta_data
    cs.mask_array_I05(LOW=220, HIGH=270)
    cs.mask_using_I01_percentile(30)
    # cs.mask_thin_cirrus(50)
    try:
        gt, i4, r2 = cs.gt_piece_percentile(plot=True, show=False,
                                        save_fig=f"{cs.metadata['NAME']}_{cs.metadata['ISO_TIME'].strftime('%Y%m%d%H')}_IMAGE.png")
    except Exception:
        continue
    # fetch future aerosol
    point_row = best_track_df.loc[(best_track_df["NAME"] == cs.metadata["NAME"]) & (
            (cs.metadata["ISO_TIME"] - best_track_df["ISO_TIME"]) <= timedelta(hours=3)) & (
                                          (cs.metadata["ISO_TIME"] - best_track_df["ISO_TIME"]) > timedelta(seconds=1))]
    point_past = best_track_df.loc[point_row.index - 8]

    LAT_PAST = point_past["USA_LAT"].item()
    LON_PAST = point_past["USA_LON"].item()

    year = point_past.iloc[0]["ISO_TIME"].year
    dayofyear = point_past.iloc[0]["ISO_TIME"].to_pydatetime().timetuple().tm_yday
    try:
        modis = AerosolImageMODIS.get_aerosol(year, dayofyear)
        aerosol = modis.get_mean_in_region(LAT_PAST, LON_PAST, 6, 6)
    except FileNotFoundError:
        aerosol = np.nan



    processed_df = processed_df.append(
        {"gt": gt.value, "i4": i4.value, "r2": r2, "aod": aerosol, "basin": point_past["BASIN"].item()},
        ignore_index=True)
print(len(processed_df))
processed_df.to_csv("out.csv")
