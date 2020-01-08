import os
import pickle
import dask
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.widgets import RectangleSelector
from pyresample import create_area_def
from satpy import Scene

from SubImage import SubImage, cubic, quadratic
from fetch_file import get_data

DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY","data")
DEFAULT_MARGIN = 0.5
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


def toggle_selector(event):
    if event.key in ['B', 'b'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


def get_eye(start_point, end_point, **kwargs):
    lat_0, lat_1 = start_point["USA_LAT"], end_point["USA_LAT"]
    lon_0, lon_1 = start_point["USA_LON"], end_point["USA_LON"]
    vel_lon_0, vel_lon_1 = start_point["STORM_SPEED"] * np.sin(start_point["STORM_DIR"] * np.pi / 180) / 60, end_point[
        "STORM_SPEED"] * np.sin(
        end_point["STORM_DIR"] * np.pi / 180) / 60
    vel_lat_0, vel_lat_1 = start_point["STORM_SPEED"] * np.cos(start_point["STORM_DIR"] * np.pi / 180) / 60, end_point[
        "STORM_SPEED"] * np.cos(
        end_point["STORM_DIR"] * np.pi / 180) / 60
    avgrmw_nm = (start_point["USA_RMW"] + end_point["USA_RMW"]) / 2
    avgrmw_deg = avgrmw_nm / 60
    dayOrNight = kwargs.get("dayOrNight", "DNB")
    try:
        files, urls = get_data(DATA_DIRECTORY, start_point["ISO_TIME"].to_pydatetime(),
                               end_point["ISO_TIME"].to_pydatetime(),
                               north=max(lat_0, lat_1) + DEFAULT_MARGIN,
                               south=min(lat_0, lat_1) - DEFAULT_MARGIN, east=wrap(max(lon_0, lon_1) + DEFAULT_MARGIN),
                               west=wrap(min(lon_0, lon_1) - DEFAULT_MARGIN),
                               dayOrNight=dayOrNight)
    except FileNotFoundError:
        return None
    raw_scene = Scene(filenames=files, reader="viirs_l1b")
    raw_scene.load(["I04", "I05", "i_lat", "i_lon"])

    delta_time = raw_scene.start_time - start_point["ISO_TIME"].to_pydatetime()
    lat_int = cubic(delta_time.seconds / 3600, spline_d(lat_1 - lat_0, vel_lat_1 - vel_lat_0, 3, vel_lat_0),
                    spline_c(lat_1 - lat_0, vel_lat_1 - vel_lat_0, 3, vel_lat_0), vel_lat_0, lat_0)
    lon_int = cubic(delta_time.seconds / 3600, spline_d(lon_1 - lon_0, vel_lon_1 - vel_lon_0, 3, vel_lon_0),
                    spline_c(lon_1 - lon_0, vel_lon_1 - vel_lon_0, 3, vel_lon_0), vel_lon_0, lon_0)

    interpolated_w_max = delta_time.seconds * (end_point["USA_WIND"] - start_point["USA_WIND"]) / (3 * 3600) + \
                         start_point["USA_WIND"]

    area = create_area_def("eye_area",
                           {"proj": "lcc", "ellps": "WGS84", "lat_0": lat_int, "lon_0": lon_int,
                            "lat_1": lat_int},
                           resolution=RESOLUTION_DEF, units="degrees",
                           area_extent=[lon_int - 2 * avgrmw_deg, lat_int - 2 * avgrmw_deg,
                                        lon_int + 2 * avgrmw_deg, lat_int + 2 * avgrmw_deg]
                           )
    tmp_scene = raw_scene.resample(area)
    centered_lon, centered_lat = area.get_lonlat(
        *np.unravel_index(tmp_scene["I05"].values.argmax(), tmp_scene["I05"].shape))
    recentered_area = create_area_def("better_eye_area",
                                      {"proj": "lcc", "ellps": "WGS84", "lat_0": centered_lat, "lon_0": centered_lon,
                                       "lat_1": lat_int}, units="degrees", resolution=RESOLUTION_DEF, area_extent=[
            centered_lon - 2 * avgrmw_deg, centered_lat - 2 * avgrmw_deg,
            centered_lon + 2 * avgrmw_deg, centered_lat + 2 * avgrmw_deg
        ])
    new_scene = raw_scene.resample(recentered_area)
    return CycloneImage(new_scene, center=(centered_lat, centered_lon), urls=urls, rmw=avgrmw_nm * NM_TO_M,
                        margin=2 * avgrmw_deg,
                        day_or_night=dayOrNight, max_wind=interpolated_w_max, **kwargs)


def get_eye_legacy(start_point, end_point, **kwargs):
    lat = start_point["LAT"], end_point["LAT"]
    lon = start_point["LON"], end_point["LON"]
    avgrmw_nm = (start_point["USA_RMW"] + end_point["USA_RMW"]) / 2
    avgrmw_deg = avgrmw_nm / 60
    dayOrNight = kwargs.get("dayOrNight", "DNB")
    try:
        files, urls = get_data(DATA_DIRECTORY, start_point["ISO_TIME"].to_pydatetime(),
                               end_point["ISO_TIME"].to_pydatetime(),
                               north=max(lat) + DEFAULT_MARGIN,
                               south=min(lat) - DEFAULT_MARGIN, east=wrap(max(lon) + DEFAULT_MARGIN),
                               west=wrap(min(lon) - DEFAULT_MARGIN),
                               dayOrNight=dayOrNight)
    except FileNotFoundError:
        return None
    raw_scene = Scene(filenames=files, reader="viirs_l1b")
    raw_scene.load(["I04", "I05", "i_lat", "i_lon"])

    delta_time = raw_scene.start_time - start_point["ISO_TIME"].to_pydatetime()
    frac = delta_time.seconds / (3 * 3600)
    lat_int = (lat[1] - lat[0]) * frac + lat[0]
    lon_int = (lon[1] - lon[0]) * frac + lon[0]

    area = create_area_def("eye_area",
                           {"proj": "lcc", "ellps": "WGS84", "lat_0": lat_int, "lon_0": lon_int,
                            "lat_1": lat_int},
                           resolution=RESOLUTION_DEF, units="degrees",
                           area_extent=[lon_int - 2 * avgrmw_deg, lat_int - 2 * avgrmw_deg,
                                        lon_int + 2 * avgrmw_deg, lat_int + 2 * avgrmw_deg]
                           )
    tmp_scene = raw_scene.resample(area)
    centered_lon, centered_lat = area.get_lonlat(
        *np.unravel_index(tmp_scene["I05"].values.argmax(), tmp_scene["I05"].shape))
    recentered_area = create_area_def("better_eye_area",
                                      {"proj": "lcc", "ellps": "WGS84", "lat_0": centered_lat, "lon_0": centered_lon,
                                       "lat_1": lat_int}, units="degrees", resolution=RESOLUTION_DEF, area_extent=[
            centered_lon - 2 * avgrmw_deg, centered_lat - 2 * avgrmw_deg,
            centered_lon + 2 * avgrmw_deg, centered_lat + 2 * avgrmw_deg
        ])
    new_scene = raw_scene.resample(recentered_area)
    return CycloneImage(new_scene, center=(centered_lat, centered_lon), urls=urls, rmw=avgrmw_nm * NM_TO_M,
                        margin=2 * avgrmw_deg,
                        day_or_night=dayOrNight, **kwargs)


class CycloneImage:

    @staticmethod
    def load_cyclone_image(fpath):
        with open(fpath, "rb") as file:
            ci = pickle.load(file)
        assert isinstance(ci, CycloneImage)
        ci.core_scene.load(["I05", "I04", "i_lat", "i_lon"])
        if not hasattr(ci, "I04"):
            ci.I04 = ci.core_scene["I04"].values
        if not hasattr(ci, "I05"):
            ci.I05 = ci.core_scene["I04"].values
        if not hasattr(ci, "pixel_x"):
            ci.pixel_x = ci.core_scene["I04"].area.pixel_size_x
        if not hasattr(ci, "pixel_y"):
            ci.pixel_y = ci.core_scene["I04"].area.pixel_size_y
        if not hasattr(ci, "rects"):
            ci.rects = {}
        if not hasattr(ci, "width"):
            ci.width = ci.I04.shape[0]
        if not hasattr(ci, "height"):
            ci.height = ci.I04.shape[1]
        return ci

    def __init__(self, core_scene=None, center=None, **kwargs):
        """
        Initialise a CycloneImage object
        :param year: year of data
        :param month: month of data (0-12
        :param day: day of month of data
        :param center: 2D tuple storing the latitude and longitude (est) of the tropical cyclone
        :param margin: 2D storing the latitude and longitude padding to sample and search for imags
        :param day_or_night: Use day (D), night (N) or Day & Night data (DNB)
        """
        if core_scene is not None:
            self.core_scene = core_scene
            self.core_scene.load(["I05", "I04", "i_lat", "i_lon"])
            self.I04 = self.core_scene["I04"].values
            self.I05 = self.core_scene["I05"].values
            self.pixel_x = self.core_scene["I04"].area.pixel_size_x
            self.pixel_y = self.core_scene["I05"].area.pixel_size_y
            self.width = self.I04.shape[0]
            self.height = self.I04.shape[1]
        else:
            raise ValueError("You must provide either a Scene object or a filepath to a scene object")
        self.rects = {}
        self.gt = []
        self.center = center
        self.margin = DEFAULT_MARGIN
        self.day_or_night = "DNB"
        self.name = "UNKNOWN"
        for key, val in kwargs.items():
            self.__dict__[key] = val

    @property
    def is_complete(self):
        return not np.isnan(self.I04).any()

    def save_object(self):
        file_name = f"{DATA_DIRECTORY}/proc/CORE_{self.name}_{self.core_scene.start_time.strftime('%Y_%m_%d__%H_%M')}.pickle"
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    def plot_globe(self, band="I04", proj="lcc", sf=1.0):
        files = get_data(DATA_DIRECTORY, self.core_scene.start_time, self.core_scene.end_time, self.center,
                         (self.margin, self.margin), self.day_or_night)
        raw_scene = Scene(filenames=files, reader="viirs_l1b")
        area = raw_scene[band].attrs["area"].compute_optimal_bb_area(
            {"proj": proj, "lat_0": self.center[0], "lon_0": self.center[1], "lat_1": 10, "lat_2": 20}
        )
        corrected_scene = raw_scene.resample(area)  # resample image to projection
        crs = corrected_scene[band].attrs["area"].to_cartopy_crs()
        ax = plt.axes(projection=crs)
        # Cartopy methods
        ax.coastlines()
        ax.gridlines()
        ax.set_global()
        plt.imshow(corrected_scene[band], transform=crs,
                   extent=(
                       crs.bounds[0] * sf, crs.bounds[1] * sf, crs.bounds[2] * sf,
                       crs.bounds[3] * sf),
                   origin="upper")
        cb = plt.colorbar()
        cb.set_label("Kelvin (K)")
        plt.show()

    def box_rect(self,keyname):
        inst = self

        def select_callback(eclick, erelease):
            x_min, y_min = min(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata)
            x_max, y_max = max(eclick.xdata, erelease.xdata), max(eclick.ydata, erelease.ydata)

            inst.new_rect(keyname, ((x_min + x_max) / 2, (y_min + y_max) / 2), x_max - x_min, y_max - y_min)
            inst.draw_rect(keyname,fit=True)

        fig, ax = plt.subplots()
        im = ax.imshow(self.I05, origin="upper",
                       extent=[-self.pixel_x * self.I04.shape[0] * 0.5,
                               self.pixel_x * self.I04.shape[0] * 0.5,
                               -self.pixel_y * self.I04.shape[1] * 0.5,
                               self.pixel_y * self.I04.shape[1] * 0.5])
        ax.set_title(
            f"{self.name} on {self.core_scene.start_time.strftime('%Y-%m-%d')} Cat {int(self.cat)} \n Pixel Resolution:{round(self.pixel_x)} meters per pixel")
        cb = plt.colorbar(im)
        cb.set_label("Kelvin (K)")
        toggle_selector.RS = RectangleSelector(
            ax, select_callback, drawtype="box", useblit=False,
            button=[1, 3], interactive=True, minspanx=5, minspany=5, spancoords="pixels")
        plt.connect("key_press_event", toggle_selector)
        plt.show()

    def draw_eye(self, band="I04"):
        try:
            fig, ax = plt.subplots()
            im = ax.imshow(self.__dict__[band], origin="upper",
                           extent=[-self.pixel_x * self.__dict__[band].shape[0] * 0.5,
                                   self.pixel_x * self.__dict__[band].shape[0] * 0.5,
                                   -self.pixel_y * self.__dict__[band].shape[1] * 0.5,
                                   self.pixel_y * self.__dict__[band].shape[1] * 0.5])
            ax.set_title(
                f"{self.name} on {self.core_scene.start_time.strftime('%Y-%m-%d')} Cat {int(self.cat)} \n Pixel Resolution:{round(self.pixel_x)} meters per pixel\nBand:{band}")
            cb = plt.colorbar(im)
            cb.set_label("Kelvin (K)")
            plt.show()
        except (KeyError, AttributeError):
            fig, ax = plt.subplots()
            self.core_scene[band].plot.imshow()
            ax.set_title(
                f"{self.name} on {self.core_scene.start_time.strftime('%Y-%m-%d')} Cat {int(self.cat)} \n Pixel Resolution:{round(self.core_scene[band].area.pixel_size_x)} meters per pixel\nBand:{band}")
            plt.show()

    def show_where_pixels_are(self, key, **kwargs):
        rect = self.rects[key]
        fig, ax = plt.subplots()
        ax.scatter(rect.i04_flat, rect.i05_flat, s=kwargs.get("s", 0.25))

        def select_callback(eclick, erelease):
            i4min, i4max = min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata)
            i5min, i5max = min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata)
            selected_points = np.argwhere(np.logical_and(np.logical_and(i4min < rect.i04, rect.i04 < i4max),
                                                         np.logical_and(i5min < rect.i05, rect.i05 < i5max)))

            fig, ax = plt.subplots()
            ax.imshow(rect.i04)
            # These are flipped due to the way the plotting is done
            ax.scatter([p[1] for p in selected_points], [p[0] for p in selected_points], s=0.25, edgecolors="r",
                       facecolors="none")
            plt.show()

        bottom, top = plt.ylim()
        left, right = plt.xlim()
        x = np.linspace(min(rect.i05_flat), max(rect.i05_flat), 100)
        gt, gt_err, params = rect.curve_fit(cubic)
        self.gt = [gt, gt_err]
        plt.plot([cubic(x_i, *params) for x_i in x], x, 'g-', label="Curve fit")
        plt.hlines(gt, xmin=left, xmax=right, colors="r")
        plt.legend()

        toggle_selector.RS = RectangleSelector(
            ax, select_callback, drawtype="box", useblit=False,
            button=[1, 3], interactive=True, minspanx=5, minspany=5, spancoords="pixels")

        ax.set_ylim([bottom, top])
        ax.set_xlim([left, right])
        ax.invert_xaxis()
        ax.invert_yaxis()

        plt.connect('key_press_event', toggle_selector)
        plt.show()

    def new_rect(self, key, center, w, h, **kwargs):
        """
        :param center: center coordinates of the cyclone in meter offset from the center of the core image
        :param w: width in meters
        :param h: height in meters
        :param plot: should we plot
        :param filename_idx: filename index
        :param save: save figure
        :param kwargs: additional arguments
        :return: i04_splice, i05_splice of selected rectangle
        """
        try:
            area = self.core_scene["I04"].area
            bottom_left = area.get_xy_from_proj_coords(center[0] - w / 2, center[1] - h / 2)
            top_right = area.get_xy_from_proj_coords(center[0] + w / 2, center[1] + h / 2)
            i04_splice = self.I04[top_right[1]:bottom_left[1], bottom_left[0]:top_right[0]]
            i05_splice = self.I05[top_right[1]:bottom_left[1], bottom_left[0]:top_right[0]]
        except AttributeError:
            splice = self.core_scene.crop(
                xy_bbox=[center[0] - w / 2, center[1] - h / 2, center[0] + w / 2, center[1] + h / 2])
            i04_splice = splice["I04"].data.flatten()
            i05_splice = splice["I05"].data.flatten()
        self.rects[key] = SubImage(i04_splice, i05_splice, w, h, center)
        return self.rects[key]

    def get_gt_and_intensity(self, key, mode="median", plot=False):
        sub_img = self.rects[key]
        gt, gt_err, params = sub_img.curve_fit(cubic)
        self.gt = [gt, gt_err]
        cat = self.cat
        start_intensity = 0  # self.__dict__["start_intensity"]
        end_intensity = 0  # self.__dict__["end_intensity"]
        basin = self.basin
        return gt, cat, basin, self.max_wind

    def draw_rect(self, key, save=False, fit=False, **kwargs):
        rect = self.rects[key]
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter(rect.i04.flatten(), rect.i05.flatten(), s=kwargs.get("s", 0.25))
        bottom, top = plt.ylim()
        left, right = plt.xlim()
        if fit:
            try:
                x = np.linspace(min(rect.i05_flat), max(rect.i05_flat), 100)
                gt, gt_err, params = rect.curve_fit(cubic)
                self.gt = [gt, gt_err]
                plt.plot([cubic(x_i, *params) for x_i in x], x, 'g-', label="Curve fit")
                plt.hlines(gt, xmin=left, xmax=right, colors="r")
                plt.legend()
            except TypeError:
                pass
        plt.gca().set_ylim([bottom, top])
        plt.gca().set_xlim([left, right])
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.ylabel("Cloud Top Temperature (K)")
        plt.xlabel("I4 band reflectance (K)")
        plt.subplot(1, 2, 2)
        plt.imshow(self.I05, origin="upper",
                   extent=[-self.pixel_x * self.I05.shape[0] * 0.5,
                           self.pixel_x * self.I05.shape[0] * 0.5,
                           -self.pixel_y * self.I05.shape[1] * 0.5,
                           self.pixel_y * self.I05.shape[1] * 0.5])
        plt.gca().add_patch(
            Rectangle((rect.center[0] - rect.width / 2, rect.center[1] - rect.height / 2), rect.width, rect.height,
                      linewidth=1, edgecolor="r", facecolor="none"))
        cb = plt.colorbar()
        cb.set_label("Kelvin (K)")
        plt.title(f"{self.name} on {self.core_scene.start_time.strftime('%Y-%m-%d')} Cat {int(self.cat)}")
        if save is True:
            plt.savefig(
                f"Images/{self.core_scene.start_time.strftime('%Y-%m-%d')}Cat{int(self.cat)}({key}).png")
        else:
            plt.show()

    def find_eye(self, band='I04'):
        if band == 'I04':
            max_band_array = self.I04
        elif band == 'I05':
            max_band_array = self.I05

        hot_point = np.amax(max_band_array)
        cold_point = np.amin(max_band_array)
        threshold = (hot_point - cold_point) / 3

        hot_point_ind = np.unravel_index(np.argmax(max_band_array, axis=None), max_band_array.shape)

        if hot_point_ind[0] == 0:
            top_y = 0
        elif hot_point_ind[1] == 0:
            left_x = 0

        for y in range(0, hot_point_ind[0]):
            if max_band_array[hot_point_ind[0] - y, hot_point_ind[1]] < max_band_array[hot_point_ind] - threshold:
                top_y = hot_point_ind[0] - y
                break
            elif y == hot_point_ind[0] - 1:
                top_y = 0
        for y in range(0, len(max_band_array) - hot_point_ind[0]):
            if max_band_array[hot_point_ind[0] + y, hot_point_ind[1]] < max_band_array[hot_point_ind] - threshold:
                bot_y = hot_point_ind[0] + y
                break
            elif y == len(max_band_array) - hot_point_ind[0] - 1:
                bot_y = len(max_band_array)
        for x in range(0, hot_point_ind[1]):
            if max_band_array[hot_point_ind[0], hot_point_ind[1] - x] < max_band_array[hot_point_ind] - threshold:
                left_x = hot_point_ind[1] - x
                break
            elif x == hot_point_ind[1] - 1:
                left_x = 0
        for x in range(0, len(max_band_array[0]) - hot_point_ind[1]):
            if max_band_array[hot_point_ind[0], hot_point_ind[1] + x] < max_band_array[hot_point_ind] - threshold:
                right_x = hot_point_ind[1] + x
                break
            elif x == len(max_band_array[0]) - hot_point_ind[1] - 1:
                right_x = len(max_band_array[0])

        return hot_point_ind, right_x, left_x, top_y, bot_y
