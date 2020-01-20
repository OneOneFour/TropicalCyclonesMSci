import os

import numpy as np
from pyresample import create_area_def
from satpy import Scene
from shapely import geometry

from CycloneSnapshot import CycloneSnapshot
from fetch_file import get_data
import matplotlib.patches as patches

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

    def __init__(self, scene, metadata):
        self.scene = scene
        self.metadata = metadata
        self.scene.load(["I05", "I04", "i_lat", "i_lon", "i_satellite_azimuth_angle"])
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
            ax.add_geometries([box], crs=PlateCarree(), edgecolor="k",facecolor="none")

        cb = plt.colorbar(im)
        cb.set_label("Kelvin (K)")
        plt.show()

    def draw_eye(self):
        return self.draw_rectangle((self.metadata["USA_LAT"], self.metadata["USA_LON"]),
                                   4 * self.metadata["USA_RMW"] * NM_TO_M, self.metadata["USA_RMW"] * 4 * NM_TO_M)

    def draw_rectangle(self, center, width, height):
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
                             sub_scene["i_satellite_azimuth_angle"].values.mean(), self.metadata)
        cs.meta_data["RECT_BLAT"] = center[0] - latitude_circle / 2
        cs.meta_data["RECT_BLON"] = center[1] - longitude_circle / 2
        cs.meta_data["RECT_W"] = longitude_circle
        cs.meta_data["RECT_H"] = latitude_circle
        self.rects.append(cs)
        return cs

    def show_fitted_pixels(self):
        i05_flat = self.I05.flatten()
        i04_flat = self.I04.flatten()
        x_i05 = np.arange(MIN_CUTOFF, 273, 1)
        if len(x_i05) < 1:
            return
        y_i04 = np.array([0] * len(x_i05))
        fig, axs = plt.subplots(1,2)
        im = axs[1].imshow(self.I05)
        plt.colorbar(im)
        axs[1].set_title("%s %s" % (self.name, self.core_scene.start_time.strftime('%Y-%m-%d')))
        axs[0].scatter(i04_flat, i05_flat, s=0.25)
        for i, x in enumerate(x_i05):
            vals = i04_flat[np.where(np.logical_and(i05_flat > (x - 0.5), i05_flat < (x + 0.5)))]
            vals_5_min = []
            vals_5_min_i05val = []
            if len(vals) < 1:
                continue
            percent_range = int(np.ceil(len(vals) * 0.075))
            if x > 235:  # Takes minimum values lower than theoretical min gt and v.v.
                for j in range(percent_range):
                    if len(vals) == 0:      # Not all values of i05 will have 5 i04 values
                        break

                    vals_5_min.append(min(vals))
                    i05s_with_same_i04 = i05_flat[np.where(i04_flat == min(vals))]
                    for i05 in i05s_with_same_i04:
                        if x - 0.5 < i05 < x + 0.5:
                            vals_5_min_i05val.append(i05)
                            if len(vals_5_min_i05val) > j:
                                break

                    vals = np.delete(vals, np.where(vals == min(vals)))
                y_i04[i] = np.median(vals_5_min)
                axs[0].scatter(vals_5_min, vals_5_min_i05val, color="orange", s=5)
                #rect = self.rects[key]
                #offset_x = (self.width - self.rects[key].width/self.pixel_x)/2
                #offset_y = (self.height - self.rects[key].height/self.pixel_y)/2
                for xy in range(len(vals_5_min)):
                    points = np.argwhere(np.logical_and(self.I05 == vals_5_min_i05val[xy], self.I04 == vals_5_min[xy]))
                    axs[1].scatter([p[1] for p in points], [p[0] for p in points], s=5, c="red")
            else:
                increasing_range = int(np.ceil(len(vals) * (0.075 + (235-x)*0.025)))
                for j in range(percent_range):
                    if len(vals) == 0:      # Not all values of i05 will have 5 i04 values
                        break
                    vals.sort()
                    idx = int((235-x)*0.025*len(vals) + j)          # changes idx to shift 2.5% range every x value
                    vals_5_min.append(vals[idx])
                    i05s_with_same_i04 = i05_flat[np.where(i04_flat == vals[idx])]
                    for i05 in i05s_with_same_i04:
                        if x - 0.5 < i05 < x + 0.5:
                            vals_5_min_i05val.append(i05)
                            if len(vals_5_min_i05val) > j:
                                break

                y_i04[i] = np.median(vals_5_min)
                axs[0].scatter(vals_5_min, vals_5_min_i05val, color="black", s=5)
                # rect = self.rects[key]
                # offset_x = (self.width - self.rects[key].width / self.pixel_x) / 2
                # offset_y = (self.height - self.rects[key].height / self.pixel_y) / 2
                for xy in range(len(vals_5_min)):
                    points = np.argwhere(np.logical_and(self.I05 == vals_5_min_i05val[xy], self.I04 == vals_5_min[xy]))
                    axs[1].scatter([p[1] for p in points], [p[0] for p in points], s=5, c="black")

        zero_args = np.where(y_i04 == 0)
        x_i05 = np.delete(x_i05, zero_args)
        y_i04 = np.delete(y_i04, zero_args)

        params, cov = sp.curve_fit(cubic, x_i05, y_i04, absolute_sigma=True)

        xvalues = np.arange(min(x_i05), max(x_i05), 1)
        yvalues = cubic(xvalues, *params)

        gt_ve = (-params[1] + np.sqrt(params[1] ** 2 - 3 * params[0] * params[2])) / (3 * params[0])
        if np.iscomplex(gt_ve) or min(x_i05) > gt_ve > max(x_i05):
            return
        self.gt = [gt_ve]

        axs[0].plot(yvalues, xvalues, color="r")
        axs[0].invert_xaxis()
        axs[0].invert_yaxis()
        if 300 > gt_ve > 200:
            axs[0].axhline(gt_ve, color="r")
        plt.show()
        return gt_ve

    def half_eye(self):
        I05_flat = self.I05.flatten()
        I04_flat = self.I04.flatten()
        rows = len(self.I05)
        columns = len(self.I05[0])
        half_mask = np.zeros(np.shape(self.I05))
        for i in range(int(rows/2)):
            for j in range(int(columns)):
                half_mask[i][j] = 1
        masked_I05 = np.ma.masked_outside(self.I05, 220, 270)
        # masked_I05 = np.ma.masked_array(masked_I05, mask=half_mask)
        masked_I04 = self.I04
        masked_rows = len(masked_I05)
        masked_columns = len(masked_I05[0])
        print(masked_rows, masked_columns)
        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(self.I04, self.I05, s=0.25)
        axs[0].scatter(masked_I04, masked_I05, s=0.5, c="black")
        axs[1].imshow(self.I05)
        axs[1].imshow(masked_I05, cmap="Dark2")
        axs[0].invert_xaxis()
        axs[0].invert_yaxis()
        plt.show()