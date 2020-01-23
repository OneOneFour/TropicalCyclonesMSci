import os

import numpy as np
from pyresample import create_area_def
from satpy import Scene
from shapely import geometry

from CycloneSnapshot import CycloneSnapshot
from fetch_file import get_data

DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", "C:/Users/tpklo/Documents/MSciNonCloud/Data")
DEFAULT_MARGIN = 0.
RESOLUTION_DEF = (3.75 / 6371) * 2 * np.pi
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
                           metadata)


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

    def __init__(self, scene: Scene, metadata: dict):
        self.scene = scene
        self.metadata = metadata
        self.scene.load(["I05", "I04", "I01", "M09", "i_lat", "i_lon", "i_satellite_azimuth_angle"])
        self.scene = self.scene.resample(resampler="nearest")
        self.lat = metadata["USA_LAT"]
        self.lon = metadata["USA_LON"]
        self.rects = []
        self.draw_eye()

    def plot_globe(self, band="I04"):
        area = self.scene[band].attrs["area"].compute_optimal_bb_area(
            {"proj": "lcc", "lat_0": self.lat, "lon_0": self.lon, "lat_1": self.lat}
        )
        corrected_scene = self.scene.resample(area)  # resample image to projection
        crs = corrected_scene[band].attrs["area"].to_cartopy_crs()
        from cartopy.crs import PlateCarree
        ax = plt.axes(projection=crs)
        # Cartopy methods
        ax.coastlines()
        ax.gridlines()
        ax.set_global()
        im = ax.imshow(corrected_scene[band], transform=crs,
                       extent=(
                           crs.bounds[0], crs.bounds[1], crs.bounds[2],
                           crs.bounds[3]),
                       origin="upper")

        for a in self.rects:
            box = geometry.box(minx=a.meta_data["RECT_BLON"], miny=a.meta_data["RECT_BLAT"],
                               maxx=a.meta_data["RECT_BLON"] + a.meta_data["RECT_W"],
                               maxy=a.meta_data["RECT_BLAT"] + a.meta_data["RECT_H"])
            ax.add_geometries([box], crs=PlateCarree(), edgecolor="k", facecolor="none")

        cb = plt.colorbar(im)
        cb.set_label("Kelvin (K)")
        plt.show()

    @property
    def is_eyewall_shaded(self):
        return self.eye.is_shaded

    @property
    def eye(self) -> CycloneSnapshot:
        return self.rects[0]

    def quads_np(self, width, height, n_rows=0, n_cols=0, p_width=0, p_height=0):
        """
        :param quad_width:
        :param quad_height:
        :return:
        """
        uni_area = create_area_def("quad_a",
                                   {"proj": "lcc", "lat_0": self.lat, "lat_1": self.lat, "lon_0": self.lon},
                                   area_extent=[
                                       self.lon - width / 2, self.lat - height / 2,
                                       self.lon + width / 2, self.lat + height / 2
                                   ], units="degrees")
        self.uniform_scene = self.scene.resample(uni_area)

        I4 = self.uniform_scene["I04"].values()
        I5 = self.uniform_scene["I05"].values()
        M9 = self.uniform_scene["M09"].values()
        I1 = self.uniform_scene["I01"].values()
        AZ = self.uniform_scene["i_satellite_azimuth_angle"].values()
        if p_width == 0 and p_height == 0:
            p_width, p_height = I4.shape[0] // n_rows, I4.shape[1] // n_cols

        self.grid = list(np.zeros((n_cols, n_rows)))
        for j in range(n_rows):
            for k in range(n_cols):
                cs = CycloneSnapshot(
                    I4[j * p_width:min((j + 1) * p_width, I4.shape[0]),
                    k * p_height:min(p_height * (k + 1), I4.shape[1])],
                    I5[j * p_width:min((j + 1) * p_width, I5.shape[0]),
                    k * p_height:min(p_height * (k + 1), I5.shape[1])],
                    uni_area.pixel_size_x, uni_area.pixel_size_y,
                    AZ[j * p_width:min((j + 1) * p_width, AZ.shape[0]),
                    k * p_height:min(p_height * (k + 1), AZ.shape[1])].mean(),
                    self.metadata, M09=
                    M9[j * p_width:min((j + 1) * p_width, M9.shape[0]),
                    k * p_height:min(p_height * (k + 1), M9.shape[1])],
                    I01=I1[j * p_width:min((j + 1) * p_width, M9.shape[0]),
                        k * p_height:min(p_height * (k + 1), M9.shape[1])].mean())

                self.grid[j][k] = cs
                self.rects.append(cs)

    def draw_eye(self):
        return self.draw_rectangle((self.metadata["USA_LAT"], self.metadata["USA_LON"]),
                                   4 * self.metadata["USA_RMW"] * NM_TO_M, self.metadata["USA_RMW"] * 4 * NM_TO_M)

    def draw_rectangle_rosenfeld(self, center, p_width=96, p_height=96) -> CycloneSnapshot:
        area = create_area_def(f"({center[0]},{center[1]})",
                               {"proj": "lcc", "ellps": "WGS84", "lat_0": center[0], "lat_1": center[0],
                                "lon_0": center[1]}, width=p_width, height=p_height,
                               resolution=self.scene["I05"].attrs["resolution"], center=center)
        sub_scene = self.scene.resample(area)
        cs = CycloneSnapshot(sub_scene["I04"].values, sub_scene["I05"].values, area.pixel_size_x, area.pixel_size_y,
                             sub_scene["i_satellite_azimuth_angle"].values.mean(), self.metadata,
                             M09=sub_scene["M09"].values, I01=sub_scene["I01"].values)
        cs.meta_data["RECT_W"] = p_width * self.scene["I05"].attrs["resolution"] / (NM_TO_M * 60)
        cs.meta_data["RECT_H"] = p_height * self.scene["I05"].attrs["resolution"] / (NM_TO_M * 60)
        cs.meta_data["RECT_BLAT"] = center[0] - cs.meta_data["RECT_H"] / 2
        cs.meta_data["RECT_BLON"] = center[1] - cs.meta_data["RECT_W"] / 2
        self.rects.append(cs)
        return cs

    def draw_rectangle(self, center, width, height) -> CycloneSnapshot:
        latitude_circle = (height / R_E) * (180 / np.pi)
        longitude_circle = (width / R_E) * (180 / np.pi)
        area = create_area_def(f"({center[0]},{center[1]})",
                               {"proj": "lcc", "ellps": "WGS84", "lat_0": center[0], "lat_1": center[0],
                                "lon_0": center[1]}, units="degrees", resolution=RESOLUTION_DEF,
                               area_extent=[
                                   center[1] - longitude_circle / 2, center[0] - latitude_circle / 2,
                                   center[1] + longitude_circle / 2, center[0] + latitude_circle / 2
                               ])
        sub_scene = self.scene.resample(area)
        cs = CycloneSnapshot(sub_scene["I04"].values, sub_scene["I05"].values, area.pixel_size_x, area.pixel_size_y,
                             sub_scene["i_satellite_azimuth_angle"].values.mean(), self.metadata,
                             M09=sub_scene["M09"].values, I01=sub_scene["I01"].values)
        cs.meta_data["RECT_BLAT"] = center[0] - latitude_circle / 2
        cs.meta_data["RECT_BLON"] = center[1] - longitude_circle / 2
        cs.meta_data["RECT_W"] = longitude_circle
        cs.meta_data["RECT_H"] = latitude_circle
        self.rects.append(cs)
        return cs
