import os

import numpy as np
from pyresample import create_area_def
from satpy import Scene

from CycloneSnapshot import CycloneSnapshot
from fetch_file import get_data

DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", "../data")
DEFAULT_MARGIN = 0.
RESOLUTION_DEF = (3.71 / 6371) * 2 * np.pi
NM_TO_M = 1852

spline_c = lambda dx, dv, T, v0: (3 * (dx - T * v0) - dv * T) / (T * T)
spline_d = lambda dx, dv, T, v0: (2 * (T * v0 - dx) + dv * T) / (T * T * T)


def wrap(x):
    if x > 180:
        return x - 360
    elif x < -180:
        return x + 360
    return x


def __clamp(x, min_v, max_v):
    return max(min(x, max_v), min_v)


def zero_clamp(x):
    return __clamp(x, 0, np.inf)


def nm_to_degrees(nm):
    return nm / 60


def interpolate(start, end, t):
    from pandas import isna
    int_dict = {"ISO_TIME": start["ISO_TIME"].to_pydatetime() + t}
    frac = t.seconds / (3 * 3600)
    for k in start.keys():
        if k == "ISO_TIME":
            continue
        if isna(start) and isna(end):
            continue
        if isna(start):
            int_dict[k] = end[k]
            continue
        else:
            int_dict[k] = start[k]

        try:
            int_dict[k] += (end[k] - start[k]) * frac
        except TypeError:
            continue
    return int_dict


def get_eye(start_point, end_point):
    lat_0, lat_1 = start_point["USA_LAT"], end_point["USA_LAT"]
    lon_0, lon_1 = start_point["USA_LON"], end_point["USA_LON"]
    try:
        files, urls = get_data(DATA_DIRECTORY, start_point["ISO_TIME"].to_pydatetime(),
                               end_point["ISO_TIME"].to_pydatetime(),
                               north=max(lat_0, lat_1) + DEFAULT_MARGIN,
                               south=min(lat_0, lat_1) - DEFAULT_MARGIN, east=wrap(max(lon_0, lon_1) + DEFAULT_MARGIN),
                               west=wrap(min(lon_0, lon_1) - DEFAULT_MARGIN),
                               dayOrNight="D")
    except FileNotFoundError:
        return None
    raw_scene = Scene(filenames=files, reader="viirs_l1b")
    raw_scene.load(["I04", "I05", "i_lat", "i_lon"])
    t = raw_scene.start_time - start_point["ISO_TIME"].to_pydatetime()

    metadata = interpolate(start_point, end_point, t)
    rmw = metadata["USA_RMW"]/60
    centered_lon, centered_lat = area.get_lonlat(
        *np.unravel_index(raw_scene["I05"].values.argmax(), raw_scene["I05"].shape))
    recentered_area = create_area_def("better_eye_area",
                                      {"proj": "lcc", "ellps": "WGS84", "lat_0": centered_lat, "lon_0": centered_lon,
                                        }, units="degrees", resolution=RESOLUTION_DEF, area_extent=[
            centered_lon - 2 * rmw, centered_lat - 2 * rmw,
            centered_lon + 2 * rmw, centered_lat + 2 * rmw
        ])
    new_scene = raw_scene.resample(recentered_area)
    return CycloneSnapshot(recentered_area["I04"].values(), recentered_area["I05"].values(),metadata=metadata)


def get_entire_cyclone(start_point, end_point):
    lat_0 = start_point["USA_LAT"] + end_point["USA_LAT"] /2
    lon_0 = start_point["USA_LON"] +  end_point["USA_LON"] /2
    try:
        files,urls = get_data(DATA_DIRECTORY,start_point["ISO_TIME"].to_pydatetime(),
                              end_point["ISO_TIME"].to_pydatetime(),
                              north = lat_0 + storm_size,
                              south = lat_0 - storm_size,
                              west = lon_0 - storm_size,
                              east = lon_0 + storm_size,
                              dayOrNight="D")


class CycloneImage:
    def __init__(self, scene):
        self.scene = scene

    def plot_globe(self):
        pass
