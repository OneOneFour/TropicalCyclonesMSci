import gzip
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyresample import create_area_def
from satpy import Scene
from xarray import DataArray

from GTFit import GTFit
from fetch_file import get_data

DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", './')
CACHE_DIRECTORY = os.environ.get("CACHE_DIRECTORY")

hc = 2E-7
hcsq = 1.31E-34
ABSOLUTE_ZERO = 273.15
I4_wavelength = 11100
I5_wavelength = 3400


def planck(T, wavelength):
    return ()


def clamp(x, min_v, max_v):
    return max(min(x, max_v), min_v)


class CycloneCellFast:
    __slots__ = ["image", "i4_reflectance", "gt", "gt_i4", "r2", "xmin", "xmax", "ymin", "ymax", "__condition"]
    A = 2 / (np.log(0.6 / 0.72))
    B = 2 - A * np.log(0.72)
    from pyspectral.near_infrared_reflectance import Calculator
    calc = Calculator("Suomi-NPP", "viirs", "I4")

    def __init__(self, image: "CycloneImageFast", xmin, xmax, ymin, ymax):
        self.image = image
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.i4_reflectance = self.calc.reflectance_from_tbs(self.zenith, self.i4, self.i5) * 100
        assert not np.isnan(self.i4).all() and not np.isnan(self.i5).all()
        self.gt, self.gt_i4, self.r2 = self.glaciation_temperature_mean()

    @property
    def i4(self):
        if not hasattr(self, "__condition"):
            T, BTD_4 = GTFit(self.BTD_m, self.image.raw_grid_I5[self.ymin:self.ymax, self.xmin:self.xmax]).bin_data(
                np.percentile, bin_func_args=(25,))
            self.__condition = False
            for i in range(len(T)):
                self.__condition = self.__condition | (
                        (abs(self.image.raw_grid_I5[self.ymin:self.ymax, self.xmin:self.xmax] - T[i]) < 0.5) & (
                        self.BTD_m <= BTD_4[i]))

        return np.where(self.__condition, self.image.raw_grid_I4[self.ymin:self.ymax, self.xmin:self.xmax], np.nan)

    @property
    def i5(self):
        if not hasattr(self, "__condition"):
            T, BTD_4 = GTFit(self.BTD_m, self.image.raw_grid_I5[self.ymin:self.ymax, self.xmin:self.xmax]).bin_data(
                np.percentile, bin_func_args=(25,))
            self.__condition = False
            for i in range(len(T)):
                self.__condition = self.__condition | (
                        (abs(self.image.raw_grid_I5[self.ymin:self.ymax, self.xmin:self.xmax] - T[i]) < 0.5) & (
                        self.BTD_m <= BTD_4[i]))

        return np.where(self.__condition, self.image.raw_grid_I5[self.ymin:self.ymax, self.xmin:self.xmax], np.nan)

    # @property
    # def i4(self):
    #     return np.where(self.BTD_m <= np.nanpercentile(self.BTD_m, 50),
    #                     self.image.raw_grid_I4[self.ymin:self.ymax, self.xmin:self.xmax], np.nan)
    #
    # @property
    # def i5(self):
    #     return np.where(self.BTD_m <= np.nanpercentile(self.BTD_m, 50),
    #                     self.image.raw_grid_I5[self.ymin:self.ymax, self.xmin:self.xmax], np.nan)

    @property
    def zenith(self):
        return self.image.zenith[self.ymin:self.ymax, self.xmin:self.xmax]

    @property
    def satz(self):
        return self.image.satz[self.ymin:self.ymax, self.xmin:self.xmax]

    @property
    def raz(self):
        return self.image.raz[self.ymin:self.ymax, self.xmin:self.xmax]

    @property
    def i5_c(self):
        return self.i5 - ABSOLUTE_ZERO

    @property
    def i5_flat_c(self):
        return self.i5_flat - ABSOLUTE_ZERO

    @property
    def BTD(self):
        return self.i4_flat - (self.i5_flat)

    @property
    def BTD_ratio(self):
        return (self.i4_flat - (self.i5_flat)) / (self.i5_flat)

    @property
    def i4i5ratio(self):
        return (self.i4_flat) / (self.i5_flat)

    @property
    def good_gt(self):
        val =  (ABSOLUTE_ZERO - 45 < self.gt.value < ABSOLUTE_ZERO) and self.r2 > 0.85
        if not val:
            print(f"FALSE:{self.gt.value},{self.gt.error},{self.r2}")
        return val
    def bin_data_percentiles(self, percentiles, i4_band=None):
        i4_list, i5_list = [], []
        if i4_band is None:
            i4_band = self.i4_flat
        gt_fit = GTFit(i4_band, self.i5_flat)
        for p in percentiles:
            i5, i4 = gt_fit.bin_data(np.percentile, 1, (p,))
            i4_list.append(i4)
            i5_list.append(i5)
        return i4_list, i5_list

    def intersects(self, cell: "CycloneCellFast"):
        if cell is self:
            return True
        return cell.xmax > self.xmin and self.xmax > cell.xmin and self.ymax > cell.ymin and cell.ymax > self.ymin

    def plot(self, band="i5", fig=None, ax=None, show=True):
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(getattr(self, band), extent=[
            -0.4 * 0.5 * self.shape[1],
            0.4 * 0.5 * self.shape[1],
            -0.4 * 0.5 * self.shape[0],
            0.4 * 0.5 * self.shape[0]
        ])
        ax.set_xlabel("km")
        ax.set_ylabel("km")
        cb = plt.colorbar(im)
        ax.set_title(f"{band}")
        if band in ("i1", "m9", "i4_reflectance"):
            cb.set_label("Reflectance (%)")
        else:
            cb.set_label("Kelvin (K)")
        if show:
            plt.show()

    def plot_i5vsi4r(self):
        fig, axs = plt.subplots(1, 2)
        im2 = axs[0].imshow(self.zenith, extent=[
            -0.4 * 0.5 * self.shape[1],
            0.4 * 0.5 * self.shape[1],
            -0.4 * 0.5 * self.shape[0],
            0.4 * 0.5 * self.shape[0]
        ])
        im = axs[1].imshow(self.i4_reflectance, extent=[
            -0.4 * 0.5 * self.shape[1],
            0.4 * 0.5 * self.shape[1],
            -0.4 * 0.5 * self.shape[0],
            0.4 * 0.5 * self.shape[0]
        ])
        axs[1].set_xlabel("km")
        axs[1].set_ylabel("km")
        plt.show()

    def plot_profile(self):
        fig, ax = plt.subplots(1, 2)
        self.plot("i5", fig, ax[0], show=False)
        gt_fit = GTFit(self.i4_reflectance_flat, self.i5_flat)
        gt_fit.piecewise_fit(fig, ax[1], units="kelvin", i4_units="reflectance")
        plt.show()

    def plot_raw_profile(self):
        fig, ax = plt.subplots(1, 2)
        self.plot("i5", fig, ax[0], show=False)
        ax[1].scatter(self.i4_reflectance_flat, self.i5_flat, s=0.1, c="r")
        ax[1].set_ylabel("Temperature (K)")
        ax[1].set_xlabel("I4 Band reflectance (%)")
        ax[1].invert_xaxis()
        ax[1].invert_yaxis()
        plt.show()

    @property
    def shape(self):
        return self.i5.shape

    @property
    def i4_flat(self):
        return self.i4[~np.isnan(self.i4)].flatten()

    @property
    def i4_reflectance_flat(self):
        return self.i4_reflectance[~np.isnan(self.i4_reflectance)].flatten()

    @property
    def i5_flat(self):
        return self.i5[~np.isnan(self.i5)].flatten()

    @property
    def m9(self):
        return self.image.mask(self.image.scene["M09"])[self.ymin:self.ymax, self.xmin:self.xmax]

    @property
    def BTD_m(self):
        return self.image.BTD_m[self.ymin:self.ymax, self.xmin:self.xmax]

    @property
    def i_satellite_azimuth(self):
        return self.image.mask(self.image.scene["i_satellite_azimuth_angle"])[self.ymin:self.ymax, self.xmin:self.xmax]

    @property
    def BTD_h(self):
        return (self.image.mask(self.image.scene["I05"]) - self.image.mask(self.image.scene["M16"]))[
               self.ymin:self.ymax, self.xmin:self.xmax]

    def glaciation_temperature_percentile(self, percentiles=(5, 50, 95)):
        gt_fit = GTFit(self.i4_reflectance_flat, self.i5.flatten())
        return gt_fit.piecewise_percentile_multiple(percentiles, units="kelvin", i4_units="reflectance")

    def glaciation_temperature_mean(self):
        gt_fit = GTFit(self.i4_reflectance_flat, self.i5_flat)
        return gt_fit.piecewise_fit(units="kelvin", i4_units="reflectance")

    def re(self, nan_out_of_range=False):
        from inspect_nc import get_re
        return np.array(
            [[get_re(self.i4_reflectance[y, x], self.satz[y, x], self.zenith[y, x], np.abs(self.raz[y, x]),
                     nan_out_of_range=nan_out_of_range) for x in
              range(self.xmax - self.xmin)] for y in range(self.ymax - self.ymin)])


class CycloneImageFast:
    """
    I/O class to read and dump files to a flattened and cropped image in .nc format
    Faster than existing data
    """
    BASE_DATASETS = ["I05", "I04"]
    OPTIONAL_DATASETS = ["M09", "I01"]
    DEFAULT_CROP = 16  # 20 deg x 20 deg

    @classmethod
    def from_gzp(cls, gzp_file):
        with gzip.GzipFile(gzp_file, "r") as f:
            inst__ = pickle.load(f)
        assert isinstance(inst__, cls)
        return inst__

    @classmethod
    def from_points(cls, start_idx, start_point, end_point):
        lat_0 = (start_point["USA_LAT"] + end_point["USA_LAT"]) / 2
        lon_0 = (start_point["USA_LON"] + end_point["USA_LON"]) / 2
        north_extent = (start_point["USA_R34_NE"] + start_point["USA_R34_NW"]) / 120
        south_extent = (start_point["USA_R34_SE"] + start_point["USA_R34_SW"]) / 120
        west_extent = (start_point["USA_R34_SW"] + start_point["USA_R34_NW"]) / 120
        east_extent = (start_point["USA_R34_NE"] + start_point["USA_R34_NW"]) / 120

        files, urls = get_data(DATA_DIRECTORY, start_point["ISO_TIME"].to_pydatetime(),
                               end_point["ISO_TIME"].to_pydatetime(),
                               north=lat_0 + north_extent,
                               south=lat_0 - south_extent,
                               west=clamp(lon_0 - west_extent, -180, 180),
                               east=clamp(lon_0 + east_extent, -180, 180),
                               dayOrNight="D", include_mod=True)
        inst__ = cls()
        inst__.scene = Scene(files, reader="viirs_l1b")
        inst__.scene.load(
            ["I01", "I04", "I05", "M09", "M16", "M15", "i_solar_zenith_angle", "i_satellite_zenith_angle",
             "i_satellite_azimuth_angle",
             "i_solar_azimuth_angle"])

        inst__.__interpolate(start_point, end_point)
        inst__.scene = inst__.scene.resample()
        inst__.metadata["START_IDX"] = start_idx
        assert (inst__.eye_lon, inst__.eye_lat) in inst__.bb_area()
        inst__.crop()
        inst__.eye()
        return inst__

    @property
    def eye_lat(self):
        assert "USA_LAT" in self.metadata
        return self.metadata["USA_LAT"]

    @property
    def eye_lon(self):
        assert "USA_LON" in self.metadata
        return self.metadata["USA_LON"]

    def eye(self):
        if hasattr(self, "eye_gd"):
            return self.eye_gd
        else:
            assert "USA_RMW" in self.metadata
            xmin, ymax = self.scene.max_area().get_xy_from_lonlat(self.eye_lon - 2 * self.metadata["USA_RMW"] / 60,
                                                                  self.eye_lat - 2 * self.metadata["USA_RMW"] / 60)
            xmax, ymin = self.scene.max_area().get_xy_from_lonlat(self.eye_lon + 2 * self.metadata["USA_RMW"] / 60,
                                                                  self.eye_lat + 2 * self.metadata["USA_RMW"] / 60)
            # Recenter
            x_list, y_list = [], []
            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    if np.isnan(self.raw_grid_I5[y, x]):
                        continue
                    x_list.append(x)
                    y_list.append(y)
            xmin = min(x_list)
            xmax = max(x_list)
            ymin = min(y_list)
            ymax = max(y_list)

            self.eye_gd = CycloneCellFast(self, xmin, xmax, ymin, ymax)
            return self.eye_gd

    def bb_area(self):
        return self.scene.max_area().compute_optimal_bb_area(
            {"proj": "lcc", "lat_0": self.eye_lat, "lat_1": self.eye_lat, "lon_0": self.eye_lon})

    def mask(self, dataarray, low=220, high=280):
        return dataarray.where((self.scene["I05"] > low) & (self.scene["I05"] < high) & (
                self.scene["I01"] > 30))

    def plot(self, band="I05"):
        self.mask(self.scene[band]).plot.imshow()
        plt.show()

    def unmask(self, recalculate=True):
        """
        Remove mask.
        :param recalculate: Recalculate the raw_grid_I4/5 variables using the new mask
        :return:
        """
        del self.raw_grid_I4
        del self.raw_grid_I5
        if recalculate:
            del self.eye_gd
            del self.cells
            self.raw_grid_I4 = self.scene["I04"].values
            self.raw_grid_I5 = self.scene["I05"].values
        else:
            print("Not recalculating. raw_grid variable will be left unbound")

    def crop(self, w=DEFAULT_CROP, h=DEFAULT_CROP):
        cropped_area = create_area_def("ci_crop",
                                       {"proj": "lcc", "ellps": "WGS84", "lat_0": self.eye_lat, "lat_1": self.eye_lat,
                                        "lon_0": self.eye_lon}, center=(self.eye_lon, self.eye_lat),
                                       radius=(w / 2, h / 2), units="degrees",
                                       resolution=DataArray(400, attrs={"units": "meters"}))
        print(cropped_area.overlap_rate(self.bb_area()) * (self.bb_area().get_area() / cropped_area.get_area()))
        assert cropped_area.overlap_rate(self.bb_area()) * (self.bb_area().get_area() / cropped_area.get_area()) > 0.8
        self.scene = self.scene.resample(self.bb_area())
        self.scene = self.scene.crop(area=cropped_area)
        self.raw_grid_I5 = self.mask(self.scene["I05"]).values
        self.raw_grid_I4 = self.mask(self.scene["I04"]).values
        self.zenith = self.scene["i_solar_zenith_angle"].values
        self.satz = np.cos(np.deg2rad(self.scene["i_satellite_zenith_angle"])).values
        self.raz = np.abs(self.scene["i_satellite_azimuth_angle"] - self.scene["i_solar_azimuth_angle"]).values
        self.BTD_m = self.mask(self.scene["M15"] - self.scene["M16"]).values

    def calculate_BTDs(self):
        BTD_m_re = (self.scene["M15"] - self.scene["M16"]).where(self.scene["I05_mask"])
        T_11, BTD_4 = GTFit(BTD_m_re.values, self.raw_grid_I5).bin_data(
            np.percentile, bin_func_args=(25,))
        condition = False
        plt.plot(T_11, BTD_4)
        plt.show()
        for i in range(len(T_11)):
            condition = condition | ((abs(self.scene["I05_mask"] - T_11[i]) < 0.5) & (
                    (BTD_m_re) < BTD_4[i]))
        masked_i5 = self.scene["I05_mask"].where(condition)
        masked_i5.plot.imshow()
        plt.show()

    def get_environmental_gts(self):
        assert hasattr(self, "cells")
        return [c.gt.value for c in self.cells], [c.gt.error for c in self.cells]

    def __interpolate(self, start: dict, end: dict):
        from pandas import isna
        t = self.scene.start_time - start["ISO_TIME"].to_pydatetime()
        int_dict = {"ISO_TIME": self.scene.start_time}
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
        self.metadata = int_dict

    def draw_grid(self):
        fig, ax = plt.subplots()
        shape = self.raw_grid_I5.shape
        im = ax.imshow(self.raw_grid_I5,
                       extent=[-0.4 * 0.5 * shape[1], 0.4 * 0.5 * shape[1], -0.4 * 0.5 * shape[0],
                               0.4 * 0.5 * shape[0]])
        ax.set_ylabel("km")
        ax.set_xlabel("km")
        units_p_cell = 0.4 * 192
        for i in range(shape[0] // 192):
            for j in range(shape[1] // 192):
                ax.add_patch(plt.Rectangle((ax.get_xlim()[0] + j * units_p_cell, ax.get_ylim()[1] - i * units_p_cell),
                                           units_p_cell, units_p_cell,
                                           fill=False, lw=2, edgecolor="black"))
        plt.show()

    def environment_bin_percentiles(self, percentiles, reflectivity=True, mean=False):
        assert hasattr(self, "cells")
        i4_list = [[] for p in percentiles]
        i5_list = [[] for p in percentiles]
        for c in self.cells:
            if reflectivity:
                i4, i5 = c.bin_data_percentiles(percentiles, c.i4_reflectance_flat)
            else:
                i4, i5 = c.bin_data_percentiles(percentiles)
            for i in range(len(percentiles)):
                i4_list[i].extend(i4[i])
                i5_list[i].extend(i5[i])
        if mean:
            for i, p in enumerate(percentiles):
                i5_list[i], i4_list[i] = GTFit(i4_list[i], i5_list[i]).bin_data(np.mean)
        return i4_list, i5_list

    def generate_environmental(self, w=192, h=192):
        shape = self.raw_grid_I5.shape
        self.cells = []
        for i in range(shape[0] // h):
            for j in range(shape[1] // w):
                try:
                    ccf = CycloneCellFast(self, j * w,
                                          (j + 1) * w, i * h,
                                          (i + 1) * h)
                    if ccf.good_gt and not ccf.intersects(self.eye()):
                        self.cells.append(ccf)
                except (ValueError, RuntimeError, AssertionError):
                    continue
        return self.cells

    def write(self):
        nc_meta = {k: str(v) if isinstance(v, datetime) else v for k, v in self.metadata.items()}
        del nc_meta["NAME"]
        del nc_meta["LAT"]
        del nc_meta["LON"]
        self.scene.save_datasets(writer="cf", filename="IRMA.nc", header_attrs=nc_meta)

    def plot_eyewall_against_external(self):
        fig, ax = plt.subplots()
        i4_l, i5_l = self.eye().bin_data_percentiles((5, 50, 95))
        i4_l_ext, i5_l_ext = self.environment_bin_percentiles((5, 50, 95), mean=True)
        for i, p in enumerate((5, 50, 95)):
            ax.plot(i4_l[i], i5_l[i], label=f"{p}th percentile eyewall")
            ax.plot(i4_l_ext[i], i5_l_ext[i], label=f"{p}th percentile external")
        ax.set_xlabel("I4 Brightness Temperature (K)")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_ylabel("Cloud top temperature (K)")
        ax.legend()
        plt.show()

    def plot_eyewall_against_ext_ref(self):
        fig, ax = plt.subplots()
        ref_l, i5_l = self.eye().bin_data_percentiles((5, 50, 95), self.eye().i4_reflectance_flat)
        ref_l_ext, i5_l_ext = self.environment_bin_percentiles((5, 50, 95), reflectivity=True, mean=True)
        for i, p in enumerate((5, 50, 95)):
            ax.plot(ref_l[i], i5_l[i], label=f"{p}th percentile eyewall")
            ax.plot(ref_l_ext[i], i5_l_ext[i], label=f"{p}th percentile external")
        ax.set_xlabel("I4 reflectivity")
        ax.set_ylabel("Cloud top temperature (K)")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.legend()
        plt.show()

    def pickle(self):

        filename = f"{self.metadata['NAME']}.{self.metadata['START_IDX']}.gzp"
        with gzip.GzipFile(os.path.join(CACHE_DIRECTORY, filename), 'w') as f:
            pickle.dump(self, f)
