import os
from pathlib import Path

import numpy as np
from pyproj import Proj
from pyresample import create_area_def, AreaDefinition
from pyresample.utils import proj4_dict_to_str
from satpy import Scene
from shapely import geometry

from AerosolImage import AerosolImageMODIS
from CycloneSnapshot import CycloneSnapshot, SnapshotGrid
from fetch_file import get_data

DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", os.getcwd())
CACHE_DIRECTORY = os.environ.get("CACHE_DIRECTORY")
DEFAULT_MARGIN = 0.2
RESOLUTION_DEF = (3.75 / 6371) * 2 * np.pi
NM_TO_M = 1852
R_E = 6371000
DEG_STEP = 375 / (NM_TO_M * 60)

spline_c = lambda dx, dv, T, v0: (3 * (dx - T * v0) - dv * T) / (T * T)
spline_d = lambda dx, dv, T, v0: (2 * (T * v0 - dx) + dv * T) / (T * T * T)


def get_xy_from_lon_lat(lon, lat, area: AreaDefinition):
    """
    Extension to the pyresample method to return the x,y position of the lon lat coordinate rounded
    :return:
    """
    p = Proj(area.proj_str)
    x_m, y_m = p(lon, lat)
    upl_x = area.area_extent[0]
    upl_y = area.area_extent[3]
    xscale = (area.area_extent[2] -
              area.area_extent[0]) / float(area.width)
    # because rows direction is the opposite of y's
    yscale = (area.area_extent[1] -
              area.area_extent[3]) / float(area.height)

    x__ = int((x_m - upl_x) / xscale)
    y__ = int((y_m - upl_y) / yscale)

    if x__ < 0 or x__ >= area.width:
        print("Out of bounds -> Rounding x")
        x__ = __clamp(x__, 0, area.width - 1)
    if y__ < 0 or y__ >= area.height:
        print("Out of bounds -> Rounding y")
        y__ = __clamp(y__, 0, area.height - 1)

    return x__, y__


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
        if isna(start[k]) or isna(end[k]):
            if not isna(start[k]):
                int_dict[k] = start[k]
            elif not isna(end[k]):
                int_dict[k] = end[k]
            continue

        try:
            int_dict[k] = (end[k] - start[k]) * frac + start[k]
        except TypeError:
            int_dict[k] = start[k]
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
                           recentered_area.pixel_size_y, new_scene["i_satellite_azimuth_angle"].values,
                           metadata, b_lon=centered_lon - 2 * eye_radius, b_lat=centered_lat - 2 * eye_radius)


def get_entire_cyclone(start_point, end_point, history=None, future=None):
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
                               dayOrNight="D", include_mod=False)
    except FileNotFoundError:
        return None

    scene = Scene(filenames=files, reader="viirs_l1b")
    t = scene.start_time - start_point["ISO_TIME"].to_pydatetime()
    metadata = interpolate(start_point, end_point, t)
    # for i, h in enumerate(history):
    #     metadata[f"DELTA_SPEED_-{(len(history) - i) * 3}HR"] = metadata["USA_WIND"] - h["USA_WIND"]
    # for i, f in enumerate(future):
    #     metadata[f"DELTA_SPEED_+{(i + 1) * 3}HR"] = f["USA_WIND"] - metadata["USA_WIND"]
    #     if (i + 1) * 3 == 24:
    #         metadata["24_HRS_LAT"] = f["USA_LAT"]
    #         metadata["24_HRS_LON"] = f["USA_LON"]

    checkpath = os.path.join(CACHE_DIRECTORY,
                             f"{metadata['NAME']}.{metadata['ISO_TIME'].strftime('%Y%m%d%H%M%S')}.gpz")
    if os.path.isfile(checkpath):
        return CycloneImage.load(checkpath)
    return CycloneImage(scene, metadata, load_mod=False)


import matplotlib.pyplot as plt


class CycloneImage:
    __slots__ = ["scene", "metadata", "lon", "lat", "rects", "proj_dict"]

    @staticmethod
    def load(fpath):
        import pickle, gzip
        with gzip.GzipFile(fpath, "r") as f_pickle:
            obj = pickle.load(f_pickle)
        assert isinstance(obj, CycloneImage)
        return obj

    def __init__(self, scene: Scene, metadata: dict, load_mod=False):
        self.scene = scene
        self.metadata = metadata

        self.scene.load(["I01", "I04", "I05", "i_satellite_azimuth_angle", "i_solar_zenith_angle"])
        if load_mod:
            self.scene.load(["M09"])
        self.scene = self.scene.resample(resampler="nearest")
        self.lat = metadata["USA_LAT"]
        self.lon = metadata["USA_LON"]
        self.rects = []
        self.proj_dict = {"proj": "lcc", "lat_0": self.lat, "lon_0": self.lon, "lat_1": self.lat}
        self.bounding_snapshot()

        print("Processing Eye")
        self.draw_eye()
        self.mask(self.eye)

    @property
    def day_of_year(self):
        return self.metadata["ISO_TIME"].timetuple().tm_yday

    def mask(self, instance: CycloneSnapshot):
        instance.mask_using_I01_percentile(30)  # 1 sigma
        instance.mask_array_I05(HIGH=270, LOW=220)
        # instance.mask_thin_cirrus(50)

    def save(self):
        # Grid is not saved, dump all others
        self.rects = self.rects[:2]
        import pickle, gzip
        with gzip.GzipFile(os.path.join(CACHE_DIRECTORY,
                                        f"{self.metadata['NAME']}.{self.metadata['ISO_TIME'].strftime('%Y%m%d%H%M%S')}.gpz"),
                           'w') as f_pickle:
            pickle.dump(self, f_pickle)

    @property
    def loaded(self):
        return {k.name for k in self.scene.keys()}

    @property
    def proj_str(self):
        return proj4_dict_to_str(self.proj_dict)

    def get_dir(self) -> str:
        """
        Return directory assigned to this class instance in order to save all output files created by this instance run.
        Will be created if it does not already exist.

        :return: Directory path assigned to this class instance
        """
        dir = os.path.join(os.environ.get("OUTPUT_DIRECTORY", "./out"), self.metadata["NAME"],
                           self.metadata["ISO_TIME"].strftime("%Y-%m-%d %H-%M"))
        try:
            Path(dir).mkdir(parents=True)
        except FileExistsError:
            pass
        return dir

    def clean_dir(self):
        """
        Remove all files in directory assigned to this class instance
        :return: none
        """
        import shutil
        shutil.rmtree(self.get_dir())

    @property
    def bb(self) -> CycloneSnapshot:
        return self.rects[0]

    def bounding_snapshot(self):
        area = self.scene["I05"].attrs["area"].compute_optimal_bb_area(
            self.proj_dict
        )

        corrected_scene = self.scene.resample(area)
        bb = CycloneSnapshot(
            corrected_scene["I04"].values,
            corrected_scene["I05"].values,
            area.pixel_size_x, area.pixel_size_y,
            corrected_scene["i_satellite_azimuth_angle"].values, self.metadata,
            area.get_lonlat(area.shape[0] - 1, 0)[0],
            area.get_lonlat(area.shape[0] - 1, 0)[1],
            corrected_scene["M09"].values if "M09" in self.loaded else None,
            corrected_scene["I01"].values
        )
        self.rects.append(bb)

    def manual_gt_cycle(self):
        self.plot_globe()
        self.eye.plot()
        while True:
            try:
                lat_offset = float(input("Enter latitude offset (in degrees): "))
                lon_offset = float(input("Enter longitude offset (in degrees): "))
                width = float(input("Enter width of grid (in degrees): "))
                height = float(input("Enter height of grid (in degrees): "))
                gd = self.grid_data(self.lat + lat_offset, self.lon + lon_offset, 96, 96, width, height)
                break
            except ValueError as e:
                print("Point outside of range, please enter different set of coords")
            finally:
                self.plot_globe()
        gd.glaciation_temperature_grid()
        try:
            gt, gt_err, r2 = self.eye.gt_piece_percentile(save_fig=os.path.join(self.get_dir(), "eye_plot.png"),
                                                          show=False)
        except (ValueError, RuntimeError):
            return
        val, val_errs = gd.gt_quadrant_distribution(gt, gt_err)
        return val, val_errs

    # def get_environmental_gt(self):
    #     area = self.scene["I05"].attrs["area"].compute_optimal_bb_area(
    #         self.proj_dict
    #     )
    #     left = wrap(self.lon - 10)
    #     right = wrap(self.lon + 10)
    #     top = self.lat + 10
    #     bottom = self.lat - 10
    #     upper_left_x, upper_left_y = get_xy_from_lon_lat(left, top, area)
    #
    #     lower_right_x, lower_right_y = get_xy_from_lon_lat(right, bottom, area)
    #     environment = self.bb.add_sub_snap_edge(upper_left_x, lower_right_x, upper_left_y, lower_right_y, left, bottom)
    #     self.mask(environment)
    #     gt_env, i4_env, r2_env = environment.gt_piece_percentile(plot=True, show=True)
    #     self.eye.gt_piece_percentile(plot=True,show=True)
    #
    #     print("LOOKING AT TOTAL DISTR")
    #
    #     environment.gt_piece_percentile_multiple(percentiles=(5,50,95))
    #     self.eye.gt_piece_percentile_multiple(percentiles=(5,50,95))
    #     print(f"GT:{gt_env}\nI4:{i4_env}\nr2:{r2_env}")

    def auto_gt_cycle(self, w=25, h=25, p_w=96, p_h=96):
        print("Processing Grid")
        gd = self.grid_data_edges(self.lon - w / 2, self.lon + w / 2, self.lat + h / 2, self.lat - h / 2, p_w, p_h)
        # if not os.path.isfile(os.path.join(self.get_dir(), "image_grid.png")):
        #     self.plot_globe(band="I05", show_fig=False, save=True)
        self.bb.plot(band="I05", save_dir=os.path.join(self.get_dir(), "whole_masked_plot.png"), show=False)
        print("Calculating variables")

        gt, i4, r2 = self.eye.gt_piece_percentile(plot=True,
                                                  save_fig=os.path.join(self.get_dir(), "eye_plot_all.png"),
                                                  show=False)

        gd.set_eye_gt(gt.value, gt.error, i4.value, i4.error)
        gd.glaciation_temperature_grid(show=False, save=True)
        gd.histogram_from_eye(show=False, save=True)
        gd.vals["24HR_AOD"] = self.get_future_aerosol()
        print(f"Eye Glaciation temperature:{gt.value}pm{gt.error} with a goodness of fit of {r2}")
        gd.add_bins()
        gd.save()
        self.save()
        return gd.vals

    def plot_globe(self, band="I05", show=-1, show_fig=True, save=False):
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

        for i, a in enumerate(self.rects):
            box = geometry.box(minx=a.b_lon, miny=a.b_lat,
                               maxx=a.b_lon + a.width,
                               maxy=a.b_lat + a.height)
            if show == i:
                ax.add_geometries([box], crs=PlateCarree(), edgecolor="r", facecolor="red", alpha=0.3)
            else:
                ax.add_geometries([box], crs=PlateCarree(), edgecolor="k", facecolor="none")
        ax.set_title(
            f"{self.metadata['NAME']} on {self.metadata['ISO_TIME']}\nCategory {self.metadata['USA_SSHS']} Wind Speed:{self.metadata['USA_WIND']}kts\nSpeed: {round(self.metadata['STORM_SPEED'])}kts@{round(self.metadata['STORM_DIR'])}\u00b0\n Eye @ {round(self.lat, 2)} \u00b0N , {round(self.lon, 2)}\u00b0E")
        cb = plt.colorbar(im)
        cb.set_label("Kelvin (K)")
        if save:
            plt.savefig(os.path.join(self.get_dir(), "image_grid.png"))
            plt.close()
        if show_fig:
            plt.show()

    @property
    def is_eyewall_shaded(self):
        return self.eye.is_shaded

    @property
    def is_eyewall_gt_good(self):
        try:
            self.eye.gt_piece_percentile(plot=False, show=False)
        except (ValueError, RuntimeError):
            return False
        else:
            return True

    @property
    def eye(self) -> CycloneSnapshot:
        return self.rects[1]

    def grid_data_edges(self, left, right, top, bottom, p_width, p_height, ignore_eye=False):
        area = self.scene["I05"].attrs["area"].compute_optimal_bb_area(
            self.proj_dict
        )

        upper_left_x, upper_left_y = get_xy_from_lon_lat(left, top, area)

        lower_right_x, lower_right_y = get_xy_from_lon_lat(right, bottom, area)

        w_i = lower_right_x - upper_left_x
        h_i = lower_right_y - upper_left_y
        n_cols = w_i // p_width
        n_rows = h_i // p_height

        grid = [[0 for c in range(n_cols)] for r in range(n_rows)]

        for r in range(n_rows):
            for c in range(n_cols):
                x_i = upper_left_x + c * p_width
                y_i = upper_left_y + r * p_height
                lon, lat = area.get_lonlat(int(y_i + p_height / 2), int(x_i - p_width / 2))

                cs = self.bb.add_sub_snap_origin(x_i, y_i, p_width, p_height, lon, lat)
                self.mask(cs)
                self.rects.append(cs)
                grid[r][c] = cs
        return SnapshotGrid(grid, self)

    def grid_data(self, lat, lon, p_width, p_height, width, height):
        area = self.scene["I05"].attrs["area"].compute_optimal_bb_area(
            self.proj_dict
        )
        upper_left_x, upper_left_y = (area.get_xy_from_lonlat(lon - width / 2, lat + height / 2))
        lower_right_x, lower_right_y = area.get_xy_from_lonlat(lon + width / 2,
                                                               lat - height / 2)
        w_i = lower_right_x - upper_left_x
        h_i = lower_right_y - upper_left_y
        n_cols = w_i // p_width
        n_rows = h_i // p_height

        grid = [[0 for c in range(n_cols)] for r in range(n_rows)]

        for r in range(n_rows):
            for c in range(n_cols):
                x_i = upper_left_x + c * p_width
                y_i = upper_left_y + r * p_height
                lon, lat = area.get_lonlat(int(y_i + p_height / 2), int(x_i - p_width / 2))
                cs = self.bb.add_sub_snap_origin(x_i, y_i, p_width, p_height, lon, lat)
                cs.mask_array_I05(HIGH=290, LOW=230)
                self.rects.append(cs)
                grid[r][c] = cs
        return SnapshotGrid(grid, self)

    def draw_eye(self):
        return self.get_rect(self.metadata["USA_LAT"], self.metadata["USA_LON"],
                             4 * self.metadata["USA_RMW"] / 60, self.metadata["USA_RMW"] * 4 / 60)

    def get_future_aerosol(self):
        ai = AerosolImageMODIS.get_aerosol(self.metadata["ISO_TIME"].year, self.day_of_year)
        try:
            return ai.get_mean_in_region(self.metadata["24_HRS_LAT"], self.metadata["24_HRS_LON"], 5, 5)
        except (KeyError, FileNotFoundError):
            return 0

    def get_rect(self, lat, lon, width, height):
        area = self.scene["I05"].attrs["area"].compute_optimal_bb_area(
            self.proj_dict
        )
        bottom_left = area.get_xy_from_lonlat(lon - width / 2, lat - height / 2)
        top_right = area.get_xy_from_lonlat(lon + width / 2, lat + height / 2)
        cs = self.bb.add_sub_snap_edge(bottom_left[0], top_right[0], top_right[1], bottom_left[1]
                                       , b_lat=lat - height / 2, b_lon=lon - width / 2)
        self.rects.append(cs)
        return cs
