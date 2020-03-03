import gzip
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from pyresample import geometry
from pyresample.kd_tree import resample_nearest

from fetch_file import get_all_modis_data

MODIS_PATH = os.environ.get("MODIS_PATH", os.getcwd())


class AerosolImageMODIS:
    AEROSOL_VARIABLE = "AODc_int"
    LATITUDE = "lat"
    LONGITUDE = "lon"
    DEFAULT_PROJECTION = {"proj": "eqc", "lat_ts": 0}
    DEGREE_TO_M = 111000

    @classmethod
    def generate_pickles(cls):
        for year in os.listdir(MODIS_PATH):
            files = glob(os.path.join(MODIS_PATH, year, "new.***.c61.nc"))
            for file in files:
                day = file[-10:-7]
                AerosolImageMODIS(int(year), int(day)).save()

    @classmethod
    def get_aerosol(cls, year, day) -> "AerosolImageMODIS":
        with gzip.GzipFile(cls.path(year, day)) as fp:
            inst__ = pickle.load(fp)
        assert isinstance(inst__, cls)
        return inst__

    @staticmethod
    def path(year, day) -> str:
        return os.path.join(MODIS_PATH, str(year), f"AerosolImage.{str(day).zfill(3)}.gzp")

    @staticmethod
    def get_modis_file(year, day) -> str:
        return os.path.join(MODIS_PATH, str(year), f"new.{str(day).zfill(3)}.c61.nc")

    def __init__(self, year, day):
        self.day = day
        self.year = year
        with Dataset(self.get_modis_file(year, day)) as rootgrp:
            __raw_aod = rootgrp[self.AEROSOL_VARIABLE][0]
            lat = rootgrp[self.LATITUDE]
            lon = rootgrp[self.LONGITUDE]
            self.lat, self.lon = np.meshgrid(lat, lon)
            self.__swath = geometry.SwathDefinition(lats=self.lat, lons=self.lon)
            self.bb_area = self.__swath.compute_optimal_bb_area(self.DEFAULT_PROJECTION)
            self.aod = resample_nearest(self.__swath, __raw_aod, self.bb_area, radius_of_influence=self.DEGREE_TO_M)

    @property
    def crs(self):
        return self.bb_area.to_cartopy_crs()

    def plot(self):
        ax = plt.axes(projection=self.crs)
        ax.coastlines()
        ax.gridlines()
        ax.set_global()
        ax.imshow(self.aod, transform=self.crs, extent=self.crs.bounds, origin="upper")
        plt.show()

    def get_mean_in_region(self, lat, lon, width, height):
        top_right_y, top_right_x = self.bb_area.get_xy_from_lonlat(lon + width / 2, lat + height / 2)
        bottom_left_y, bottom_left_x = self.bb_area.get_xy_from_lonlat(lon - width / 2, lat - height / 2)
        box = self.aod[top_right_x:bottom_left_x, bottom_left_y:top_right_y:]
        return np.mean(box)

    def save(self):
        with gzip.GzipFile(self.path(self.year, self.day), "w") as fp:
            pickle.dump(self, fp)


if __name__ == "__main__":
    AerosolImageMODIS.generate_pickles()
