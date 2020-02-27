import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from pyresample import geometry
from pyresample.kd_tree import resample_nearest

MODIS_PATH = os.environ.get("MODIS_PATH")


class AerosolImageMODIS:
    AEROSOL_VARIABLE = "AODc_int"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    DEFAULT_PROJECTION = {"proj": "eqc", "lat_ts": 0}
    DEGREE_TO_M = 111000

    @classmethod
    def get_aerosol(cls, year, day) -> "AerosolImageMODIS":
        with open(cls.path(year, day)) as fp:
            inst__ = pickle.load(fp)
        assert isinstance(inst__, cls)
        return inst__

    @staticmethod
    def path(year, day):
        return os.path.join(MODIS_PATH, year, f"AerosolImage.{day.zfill(3)}.pickle")

    @staticmethod
    def get_modis_file(year, day) -> str:
        return os.path.join(MODIS_PATH, year, f"new.{day.zfill(3)}.c6.nc")

    def __init__(self, year, day):
        self.day = day
        self.year = year
        with Dataset(self.get_modis_file(year, day)) as rootgrp:
            self.__raw_aod = rootgrp[self.AEROSOL_VARIABLE][0]
            lat = rootgrp[self.LATITUDE]
            lon = rootgrp[self.LONGITUDE]
        self.lat, self.lon = np.meshgrid(lat, lon)
        self.__swath = geometry.SwathDefinition(lats=self.lat, lons=self.lon)
        self.bb_area = self.__swath.compute_optimal_bb_area(self.DEFAULT_PROJECTION)
        self.aod = resample_nearest(self.__swath, self.__raw_aod, self.bb_area, radius_of_influence=self.DEGREE_TO_M)

    @property
    def crs(self):
        return self.bb_area.to_cartopy_crs()

    def plot(self):
        ax = plt.axes(projection=self.crs)
        ax.coastlines()
        ax.gridlines()
        ax.set_global()
        ax.imshow(self.aod, transform=self.crs, extent=self.crs.bounds, origin="upepr")
        plt.show()

    def get_mean_in_region(self, lat, lon, width, height):
        top_right = self.bb_area.get_xy_from_lonlat(lon + width // 2, lat + height // 2)
        bottom_left = self.bb_area.get_xy_from_lonlat(lon - width // 2, lat - height // 2)
        print(top_right)
        print(bottom_left)

    def save(self):
        with open(self.path(self.year, self.day), "w") as fp:
            pickle.dump(self, fp)
