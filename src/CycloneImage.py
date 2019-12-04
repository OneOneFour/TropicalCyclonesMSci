import os
import pickle
import dask
from multiprocessing.pool import ThreadPool
dask.config.set(pool=ThreadPool(2))
import matplotlib.pyplot as plt
import scipy.optimize as sp
from matplotlib.patches import Rectangle
import numpy as np
from pyresample import create_area_def
from satpy import Scene

from fetch_file import get_data


DATA_DIRECTORY = "data"
DEFAULT_MARGIN = 0.5
RESOLUTION_DEF = (3.71 / 6371) * 2 * np.pi
NM_TO_M = 1852


def curve_func(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def straight_line_func(x, m, b):
    return m * x + b


def wrap(x):
    if x > 180:
        return x - 360
    elif x < -180:
        return x + 360
    return x


def __clamp(x, min_v, max_v):
    return max(min(x, max_v), min_v)


def zero_clamp(x):
    return __clamp(x,0,np.inf)


def nm_to_degrees(nm):
    return nm / 60


def get_eye(start_point, end_point, **kwargs):
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
    core_scene = raw_scene.resample(area)
    return CycloneImage(core_scene, center=(lat_int, lon_int), urls=urls, rmw=avgrmw_nm * NM_TO_M,
                        margin=2 * avgrmw_deg,
                        day_or_night=dayOrNight, **kwargs)


class CycloneImage:

    @staticmethod
    def load_cyclone_image(fpath):
        with open(fpath, "rb") as file:
            ci = pickle.load(file)
        assert isinstance(ci, CycloneImage)
        ci.core_scene.load(["I05", "I04", "i_lat", "i_lon"])
        if not hasattr(ci,"I04"):
            ci.I04 = ci.core_scene["I04"].values
        if not hasattr(ci, "I05"):
            ci.I05 = ci.core_scene["I04"].values
        if not hasattr(ci, "pixel_x"):
            ci.pixel_x = ci.core_scene["I04"].area.pixel_size_x
        if not hasattr(ci, "pixel_y"):
            ci.pixel_y = ci.core_scene["I04"].area.pixel_size_y
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
        else:
            raise ValueError("You must provide either a Scene object or a filepath to a scene object")
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

    def draw_rect(self, center, w, h, center_pixel, filename_idx, mode, **kwargs):
        try:
            ix,iy = center_pixel[0], center_pixel[1]
            iw, ih = w / self.pixel_x, h / self.pixel_y
            i04_splice = self.I04[zero_clamp(int(round(iy - ih / 2))):int(round(iy + ih / 2)), zero_clamp(int(round(ix - iw / 2))):int(round(ix + iw / 2))]
            i05_splice = self.I05[zero_clamp(int(round(iy - ih / 2))):int(round(iy + ih / 2)), zero_clamp(int(round(ix - iw / 2))):int(round(ix + iw / 2))]
        except AttributeError:
            splice = self.core_scene.crop(
                xy_bbox=[center[0] - w / 2, center[1] - h / 2, center[0] + w / 2, center[1] + h / 2])
            i04_splice = splice["I04"].data.flatten()
            i05_splice = splice["I05"].data.flatten()
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter(i04_splice.flatten(), i05_splice.flatten(), s=kwargs.get("s", 0.25))
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.ylabel("Cloud Top Temperature (K)")
        plt.xlabel("I4 band reflectance (K)")
        plt.subplot(1, 2, 2)
        plt.imshow(self.I04, origin="upper",
                   extent=[-self.pixel_x * self.I04.shape[0] * 0.5,
                           self.pixel_x * self.I04.shape[0] * 0.5,
                           -self.pixel_y * self.I04.shape[1] * 0.5,
                           self.pixel_y * self.I04.shape[1] * 0.5])
        plt.gca().add_patch(
            Rectangle((center[0] - w / 2, center[1] - h / 2), w, h, linewidth=1, edgecolor="r", facecolor="none"))
        cb = plt.colorbar()
        cb.set_label("Kelvin (K)")
        plt.title(f"{self.name} on {self.core_scene.start_time.strftime('%Y-%m-%d')} Cat {int(self.cat)}")
        if mode == "save":
            plt.savefig(
                f"Images/{self.core_scene.start_time.strftime('%Y-%m-%d')}Cat{int(self.cat)}({filename_idx}).png")

            # Code below used for pickling slices of data while knowing the graph data.
            #for pic_name in ["2013-10-19Cat4(2)", "2014-08-09Cat4(3)", "2014-10-16Cat4(4)", "2014-11-04Cat4(3)", "2015-08-30Cat4(3)"]:
            #    if pic_name == f"{self.core_scene.start_time.strftime('%Y-%m-%d')}Cat{int(self.cat)}({filename_idx})":
            #        file_name = f"proc/pic_dat_test/{self.core_scene.start_time.strftime('%Y-%m-%d')}Cat{int(self.cat)}({filename_idx}).pickle"
            #        with open(file_name, "wb") as file:
            #            pickle.dump([i04_splice, i05_splice], file)

        elif mode == "plot":
            plt.show()
        elif mode == "return":
            return i04_splice, i05_splice

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
            elif x == hot_point_ind[1]-1:
                left_x = 0
        for x in range(0, len(max_band_array[0]) - hot_point_ind[1]):
            if max_band_array[hot_point_ind[0], hot_point_ind[1] + x] < max_band_array[hot_point_ind] - threshold:
                right_x = hot_point_ind[1] + x
                break
            elif x == len(max_band_array[0]) - hot_point_ind[1] - 1:
                right_x = len(max_band_array[0])

        return hot_point_ind, right_x, left_x, top_y, bot_y


    def gt_curve_fit(self, i04flat, i05flat, mode="min", plot=False):
        idxs = np.nonzero(i05flat < 210)
        i04_fit_data = np.delete(i04flat, idxs[0])
        i05_fit_data = np.delete(i05flat, idxs[0])

        minimised_i05_fit_data = np.arange(int(min(i05_fit_data)), int(max(i05_fit_data)), 1)
        minimised_i04_fit_data = []
        num_vals_bins = []
        point_errs = 0  # Errors due to variance in I04 temps (y-data)

        for i in minimised_i05_fit_data:
            min_idxs = np.where(np.logical_and(i05_fit_data > (i - 0.5), i05_fit_data < (i + 0.5)))[0]
            i04_min_vals = []
            for idx in min_idxs:
                i04_min_vals.append(i04_fit_data[idx])
            if len(i04_min_vals) > 0:
                if mode == "mean":
                    minimised_i04_fit_data.append(np.mean(i04_min_vals))
                    mean_std = np.std(i04_min_vals) / np.sqrt(len(i04_min_vals))
                    point_errs += mean_std ** 2
                elif mode == "min":
                    minimised_i04_fit_data.append(min(i04_min_vals))
                    point_errs += 0.5 ** 2
                elif mode == "median":
                    median_i04 = np.median(i04_min_vals)
                    minimised_i04_fit_data.append(median_i04)
                    median_std = 1.253 * np.std(i04_min_vals) / np.sqrt(len(i04_min_vals))
                    point_errs += median_std ** 2
            else:
                minimised_i05_fit_data = np.delete(minimised_i05_fit_data, np.where(minimised_i05_fit_data == i)[0])

            num_vals_bins.append(
                len(i04_min_vals))  # list of number of I04 values that were in each I05 increment (for errors)

        params, cov = sp.curve_fit(curve_func, minimised_i05_fit_data, minimised_i04_fit_data)
        # perr = np.sqrt(np.diag(cov))

        xvalues = np.arange(min(minimised_i05_fit_data), max(minimised_i05_fit_data), 1)
        yvalues = curve_func(xvalues, params[0], params[1], params[2], params[3])
        i04min_ind = np.argmin(yvalues)
        gt = xvalues[i04min_ind]

        # satellite_data_error = ??
        curve_fit_err = 0.5 * abs(3 * params[0] * gt ** 2 + 2 * params[1] * gt + params[2])
        gt_err = np.sqrt(curve_fit_err ** 2 + 0.5 ** 2 + point_errs)

        if plot:
            plt.figure()
            plt.scatter(i04flat, i05flat, s=0.25)
            plt.plot(minimised_i04_fit_data, minimised_i05_fit_data, label="Data to be fitted")
            plt.plot(yvalues, xvalues, label="Line of best fit")
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            plt.legend()

        return gt, gt_err

    def gt_two_line_fit(self, i04flat, i05flat, mode="min", plot=False):
        idxs = np.nonzero(i05flat < 210)
        i04_fit_data = np.delete(i04flat, idxs[0])
        i05_fit_data = np.delete(i05flat, idxs[0])

        minimised_i05_fit_data = np.arange(int(min(i05_fit_data)), int(max(i05_fit_data)), 1)
        minimised_i04_fit_data = []
        num_vals = 0
        point_errs = 0
        found_middle = False

        for i in minimised_i05_fit_data:
            min_idxs = np.where(np.logical_and(i05_fit_data > (i - 0.5), i05_fit_data < (i + 0.5)))[0]
            i04_min_vals = []
            for idx in min_idxs:
                i04_min_vals.append(i04_fit_data[idx])

            if len(i04_min_vals) > 0:
                num_vals += len(i04_min_vals)
                if mode == "mean":
                    minimised_i04_fit_data.append(np.mean(i04_min_vals))
                    mean_std = np.std(i04_min_vals) / np.sqrt(len(i04_min_vals))
                    point_errs += mean_std ** 2
                elif mode == "min":
                    minimised_i04_fit_data.append(min(i04_min_vals))
                    point_errs += 0.5 ** 2
                elif mode == "median":
                    median_i04 = np.median(i04_min_vals)
                    minimised_i04_fit_data.append(median_i04)
                    median_std = 1.253 * np.std(i04_min_vals) / np.sqrt(len(i04_min_vals))
                    point_errs += median_std ** 2
            else:
                minimised_i05_fit_data = np.delete(minimised_i05_fit_data, np.where(minimised_i05_fit_data == i)[0])

            if not found_middle:
                if num_vals >= len(i04_fit_data) / 2:
                    mid_point = np.where(minimised_i05_fit_data == i)[0]
                    found_middle = True

        for i in mid_point:
            i05_lower_fit_data = minimised_i05_fit_data[:i]
            i05_upper_fit_data = minimised_i05_fit_data[i:]
            i04_lower_fit_data = minimised_i04_fit_data[:i]
            i04_upper_fit_data = minimised_i04_fit_data[i:]

            lower_params, lower_cov = sp.curve_fit(straight_line_func, i05_lower_fit_data, i04_lower_fit_data)
            upper_params, upper_cov = sp.curve_fit(straight_line_func, i05_upper_fit_data, i04_upper_fit_data)
            xvalues = np.arange(min(minimised_i05_fit_data), max(minimised_i05_fit_data), 1)
            lower_yvalues = straight_line_func(xvalues, lower_params[0], lower_params[1])
            upper_yvalues = straight_line_func(xvalues, upper_params[0], upper_params[1])

            fit_err_lower = abs(lower_params[0]) * 0.5
            fit_err_upper = abs(upper_params[0]) * 0.5
            fit_err_combined = np.sqrt(fit_err_upper ** 2 + fit_err_lower ** 2)
            gt_err = np.sqrt(fit_err_combined ** 2 + 2 * 0.5 ** 2 + point_errs)

            try:
                intersection_idx = np.where(lower_yvalues < upper_yvalues)[0][0]
                gt = xvalues[intersection_idx]

                if plot:
                    plt.figure()
                    plt.scatter(i04flat, i05flat, s=0.25)
                    plt.plot(i04_lower_fit_data, i05_lower_fit_data)
                    plt.plot(i04_upper_fit_data, i05_upper_fit_data)
                    plt.plot(lower_yvalues[:intersection_idx], xvalues[:intersection_idx])
                    plt.plot(upper_yvalues[intersection_idx:], xvalues[intersection_idx:])
                    plt.gca().invert_yaxis()
                    plt.gca().invert_xaxis()
            except IndexError:
                print("No min value")
                gt = 0

        return gt, gt_err

    def gt_min_i04(self, i04flat, i05flat):
        idxs = np.nonzero(i05flat < 200)
        i04_fit_data = np.delete(i04flat, idxs[0])
        i05_fit_data = np.delete(i05flat, idxs[0])
        i04min_ind = np.unravel_index(np.argmin(i04_fit_data, axis=None), i04_fit_data.shape)
        gt = i05_fit_data[i04min_ind]
        gt_err = np.std(i04flat)
        return gt, gt_err