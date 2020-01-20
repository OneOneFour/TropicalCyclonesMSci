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
R_E = 6371000

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
        if isna(start[k]) and isna(end[k]):
            continue
        if isna(start[k]):
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
    raw_scene.load(["I04", "I05", "i_lat", "i_lon", "i_satellite_azimuth_angle"])
    t = raw_scene.start_time - start_point["ISO_TIME"].to_pydatetime()

    metadata = interpolate(start_point, end_point, t)
    eye_radius = metadata["USA_RMW"] / 60
    # core_area = create_area_def("core_eye",{
    #     "proj":"lcc","ellps":"WGS84","lat_0":metadata["USA_LAT"],"lon_1":metadata["USA_LON"]},units="degrees",
    #
    # })
    first_pass = create_area_def("first_pass",
                                 {"proj": "lcc", "ellps": "WGS84", "lat_0": metadata["USA_LAT"],
                                  "lon_0": metadata["USA_LON"], "lat_1": metadata["USA_LAT"]
                                  }, units="degrees", resolution=RESOLUTION_DEF, area_extent=[
            metadata["USA_LON"] - 2 * eye_radius, metadata["USA_LAT"] - 2 * eye_radius,
            metadata["USA_LON"] + 2 * eye_radius, metadata["USA_LAT"] + 2 * eye_radius
        ])
    cropped_scene = raw_scene.resample(first_pass)
    centered_lon, centered_lat = first_pass.get_lonlat(
        *np.unravel_index(cropped_scene["I05"].values.argmax(), cropped_scene["I05"].shape))
    recentered_area = create_area_def("better_eye_area",
                                      {"proj": "lcc", "ellps": "WGS84", "lat_0": centered_lat, "lon_0": centered_lon,
                                       "lat_1": centered_lat}, units="degrees", resolution=RESOLUTION_DEF, area_extent=[
            centered_lon - 2 * eye_radius, centered_lat - 2 * eye_radius,
            centered_lon + 2 * eye_radius, centered_lat + 2 * eye_radius
        ])
    new_scene = raw_scene.resample(recentered_area)

    return CycloneSnapshot(new_scene["I04"].values, new_scene["I05"].values, recentered_area.pixel_size_x,
                           recentered_area.pixel_size_y, new_scene["i_satellite_azimuth_angle"].values.mean(),
                           metadata=metadata)


def get_entire_cyclone(start_point, end_point):
    lat_0 = (start_point["USA_LAT"] + end_point["USA_LAT"]) / 2
    lon_0 = (start_point["USA_LON"] + end_point["USA_LON"]) / 2
    north_extent = (start_point["USA_R34_NE"] + start_point["USA_R34_NW"]) / 120
    south_extent = (start_point["USA_R34_SE"] + start_point["USA_R34_SW"]) / 120
    west_extent = (start_point["USA_R34_SW"] + start_point["USA_R34_NW"]) / 120
    east_extent = (start_point["USA_R34_NE"] + start_point["USA_R34_NW"]) / 120

    try:
        files, urls = get_data(DATA_DIRECTORY, start_point["ISO_TIME"].to_pydatetime(),
                               end_point["ISO_TIME"].to_pydatetime(),
                               north=lat_0 + north_extent,
                               south=lat_0 - south_extent,
                               west=lon_0 - west_extent,
                               east=lon_0 + east_extent,
                               dayOrNight="D")
    except FileNotFoundError:
        return None

    scene = Scene(filenames=files, reader="viirs_l1b")
    t = scene.start_time - start_point["ISO_TIME"].to_pydatetime()
    metadata = interpolate(start_point, end_point, t)

    return CycloneImage(scene, metadata)


import matplotlib.pyplot as plt


class CycloneImage:

    def __init__(self, scene, metadata):
        self.scene = scene
        self.metadata = metadata
        self.scene.load(["I05", "I04", "i_lat", "i_lon"])
        self.lat = metadata["USA_LAT"]
        self.lon = metadata["USA_LON"]

    def plot_globe(self, band="I04"):
        area = self.scene[band].attrs["area"].compute_optimal_bb_area(
            {"proj": "lcc", "lat_0": self.lat, "lon_0": self.lon, "lat_1": self.lat}
        )
        corrected_scene = self.scene.resample(area)  # resample image to projection
        crs = corrected_scene[band].attrs["area"].to_cartopy_crs()
        ax = plt.axes(projection=crs)
        # Cartopy methods
        ax.coastlines()
        ax.gridlines()
        ax.set_global()
        plt.imshow(corrected_scene[band], transform=crs,
                   extent=(
                       crs.bounds[0], crs.bounds[1], crs.bounds[2],
                       crs.bounds[3]),
                   origin="upper")
        cb = plt.colorbar()
        cb.set_label("Kelvin (K)")
        plt.show()

    def draw_rectangle(self, center, width, height):
        latitude_circle = (height / R_E) * (180 / np.pi)
        longitude_circle = (width / R_E) * (180 / np.pi)
        area = create_area_def("rect",
                               {"proj": "lcc", "ellps": "WGS84", "lat_0": center[0], "lat_1": center[0],
                                "lon_0": center[1], "units": "degrees"}, units="degrees", resolution=RESOLUTION_DEF,
                               area_extent=[
                                   center[1] - longitude_circle / 2, center[0] - latitude_circle / 2,
                                   center[1] + longitude_circle / 2, center[0] + latitude_circle / 2
                               ])
        sub_scene = self.scene.resample(area)
        return CycloneSnapshot(sub_scene["I04"].values, sub_scene["I05"].values, area.pixel_size_x, area.pixel_size_y,
                               self.metadata)
