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


def clamp(x, min_v, max_v):
    return max(min(x, max_v), min_v)


class CycloneCellFast:
    __slots__ = ["i4", "i5", "gts"]

    def __init__(self, i4, i5):
        self.i4 = i4
        self.i5 = i5
        assert not np.isnan(self.i4).all() and not np.isnan(self.i5).all()
        self.gts = self.glaciation_temperature_percentile()

    @property
    def good_gt(self):
        return (-45 < self.gt.value < 10) and self.gt.error < 5 and self.r2 > 0.85

    def bin_data_percentiles(self, percentiles):
        i4_list, i5_list = [], []
        gt_fit = GTFit(self.i4.flatten(), self.i5.flatten())
        for p in percentiles:
            i5, i4 = gt_fit.bin_data(np.percentile, 1, (p,))
            i4_list.append(i4)
            i5_list.append(i5)
        return i4_list, i5_list

    def plot(self):
        plt.imshow(self.i5)
        plt.show()

    @property
    def gt(self):
        return self.gts[0][0]

    @property
    def gt_i4(self):
        return self.gts[0][1]

    @property
    def r2(self):
        return self.gts[0][2]

    def glaciation_temperature_percentile(self, percentiles=(5, 50, 95)):
        gt_fit = GTFit(self.i4.flatten(), self.i5.flatten())
        return gt_fit.piecewise_percentile_multiple(percentiles)


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
                               dayOrNight="D", include_mod=False)
        inst__ = cls()
        inst__.scene = Scene(files, reader="viirs_l1b")
        inst__.scene.load(["I04", "I05", "I01"])
        inst__.scene["I05"] -= 273.15
        inst__.__interpolate(start_point, end_point)
        inst__.metadata["START_IDX"] = start_idx
        assert (inst__.eye_lon, inst__.eye_lat) in inst__.bb_area()
        inst__.mask()
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
            xmin, ymin = self.scene.max_area().get_xy_from_lonlat(self.eye_lon - 2 * self.metadata["USA_RMW"] / 60,
                                                                  self.eye_lat - 2 * self.metadata["USA_RMW"] / 60)
            xmax, ymax = self.scene.max_area().get_xy_from_lonlat(self.eye_lon + 2 * self.metadata["USA_RMW"] / 60,
                                                                  self.eye_lat + 2 * self.metadata["USA_RMW"] / 60)
            self.eye_gd = CycloneCellFast(self.raw_grid_I4[ymax:ymin, xmin:xmax],
                                          self.raw_grid_I5[ymax:ymin, xmin:xmax])
            return self.eye_gd

    def bb_area(self):
        return self.scene.max_area().compute_optimal_bb_area(
            {"proj": "lcc", "lat_0": self.eye_lat, "lat_1": self.eye_lat, "lon_0": self.eye_lon})

    def mask(self, calculate=False):
        self.scene["I05_mask"] = self.scene["I05"].where((self.scene["I05"] < 0) & (self.scene["I05"] > -50) & (
                self.scene["I01"] > self.scene["I01"].mean() + self.scene["I01"].std()))
        self.scene["I04_mask"] = self.scene["I04"].where((self.scene["I05"] < 0) & (self.scene["I05"] > -50) & (
                self.scene["I01"] > self.scene["I01"].mean() + self.scene["I01"].std()))

        if calculate:
            self.raw_grid_I4 = self.scene["I04_mask"].values
            self.raw_grid_I5 = self.scene["I05_mask"].values

    def unmask(self, recalculate=True):
        """
        Remove mask.
        :param recalculate: Recalculate the raw_grid_I4/5 variables using the new mask
        :return:
        """
        del self.raw_grid_I4
        del self.raw_grid_I5
        del self.scene["I05_mask"]
        del self.scene["I04_mask"]
        if recalculate:
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
        self.raw_grid_I5 = self.scene["I05_mask"].values
        self.raw_grid_I4 = self.scene["I04_mask"].values
        # self.raw_grid_I1 = self.scene["I01"].values

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

    def environment_bin_percentiles(self, percentiles):
        assert hasattr(self, "cells")
        i4_list = [[] for p in percentiles]
        i5_list = [[] for p in percentiles]
        for c in self.cells:
            i4, i5 = c.bin_data_percentiles(percentiles)
            for i in range(len(percentiles)):
                i4_list[i].extend(i4[i])
                i5_list[i].extend(i5[i])
        return i4_list, i5_list

    def generate_environmental(self, w=192, h=192):
        shape = self.raw_grid_I5.shape
        self.cells = []
        for i in range(shape[0] // h):
            for j in range(shape[1] // w):
                environmental_I5 = self.raw_grid_I5[i * h:(i + 1) * h, j * w:(j + 1) * w]
                environmental_I4 = self.raw_grid_I4[i * h:(i + 1) * h, j * w:(j + 1) * w]
                if len(environmental_I5[np.isnan(environmental_I5)]) > 0.8 * w * h:
                    continue
                try:
                    ccf = CycloneCellFast(environmental_I4, environmental_I5)
                    if ccf.good_gt:
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

    def pickle(self):

        filename = f"{self.metadata['NAME']}.{self.metadata['START_IDX']}.gzp"
        with gzip.GzipFile(os.path.join(CACHE_DIRECTORY, filename), 'w') as f:
            pickle.dump(self, f)
