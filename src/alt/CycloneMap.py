from datetime import datetime

from pyresample import create_area_def
from xarray import DataArray
import matplotlib.pyplot as plt
import numpy as np
from GTFit import GTFit
from fetch_file import get_data
import os
from satpy import Scene

DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", './')


class CycloneCellFast:
    def __init__(self, i4, i5):
        self.i4 = i4
        self.i5 = i5
        self.gts = []

    @property
    def good_gt(self):
        return (-45 < self.gt.value < 0) and self.gt.error < 5 and self.r2 > 0.85

    def plot(self):
        plt.imshow(self.i4)
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

    def glaciation_temperature_percentile(self, percentiles=(5,)):
        gt_fit = GTFit(self.i4.flatten(), self.i5.flatten())

        self.gts = gt_fit.piecewise_percentile_multiple((5, 50, 95))


class CycloneImageFast:
    """
    I/O class to read and dump files to a flattened and cropped image in .nc format
    Faster than existing data
    """
    BASE_DATASETS = ["I05", "I04"]
    OPTIONAL_DATASETS = ["M09", "I01"]
    DEFAULT_CROP = 16  # 20 deg x 20 deg

    @classmethod
    def from_nc(cls, ncfile):
        inst__ = cls()
        inst__.scene = Scene([ncfile], reader="viirs_l1b")
        return inst__

    @staticmethod
    def from_cyclone_image(ci):
        pass

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
                               west=lon_0 - west_extent,
                               east=lon_0 + east_extent,
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
        return inst__

    @property
    def eye_lat(self):
        return self.metadata["USA_LAT"]

    @property
    def eye_lon(self):
        return self.metadata["USA_LON"]

    def eye(self):
        if hasattr(self, "eye_gd"):
            return self.eye_gd
        else:
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

    def mask(self):
        self.scene["I05_mask"] = self.scene["I05"].where((self.scene["I05"] < 0) & (self.scene["I05"] > -50))
        self.scene["I04_mask"] = self.scene["I04"].where((self.scene["I05"] < 0) & (self.scene["I05"] > -50))

    def crop(self, w=DEFAULT_CROP, h=DEFAULT_CROP):
        cropped_area = create_area_def("ci_crop",
                                       {"proj": "lcc", "ellps": "WGS84", "lat_0": self.eye_lat, "lat_1": self.eye_lat,
                                        "lon_0": self.eye_lon}, center=(self.eye_lon, self.eye_lat),
                                       radius=(w / 2, h / 2), units="degrees",
                                       resolution=DataArray(400, attrs={"units": "meters"}))
        self.scene = self.scene.resample(self.bb_area())
        self.scene = self.scene.crop(area=cropped_area)
        self.raw_grid_I5 = self.scene["I05_mask"].values
        self.raw_grid_I4 = self.scene["I04_mask"].values
        self.raw_grid_I1 = self.scene["I01"].values

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

    def generate_environmental(self, w=192, h=192):
        shape = self.raw_grid_I5.shape
        self.cells = []
        for i in range(shape[0] // h):
            for j in range(shape[1] // w):
                environmental_I5 = self.raw_grid_I5[i * h:(i + 1) * h, j * w:(j + 1) * w]
                environmental_I4 = self.raw_grid_I4[i * h:(i + 1) * h, j * w:(j + 1) * w]
                self.cells.append(CycloneCellFast(environmental_I4, environmental_I5))
        return self.cells

    def write(self):
        nc_meta = {k: str(v) if isinstance(v, datetime) else v for k, v in self.metadata.items()}
        del nc_meta["NAME"]
        del nc_meta["LAT"]
        del nc_meta["LON"]
        self.scene.save_datasets(writer="cf", filename="IRMA.nc", header_attrs=nc_meta)

    def pickle(self):
        import pickle
        with open("IRMA.pickle", 'wb') as f:
            pickle.dump(self, f)
