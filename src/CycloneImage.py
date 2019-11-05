from datetime import datetime

import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from pyresample import create_area_def
from satpy import Scene

from fetch_file import get_data

DATA_DIRECTORY = "data"


class CycloneImage:

    def __init__(self, year, month, day, center, margin, day_or_night="D"):
        """
        Initialise a CycloneImage object
        :param year: year of data
        :param month: month of data (0-12
        :param day: day of month of data
        :param center: 2D tuple storing the latitude and longitude (est) of the tropical cyclone
        :param margin: 2D storing the latitude and longitude padding to sample and search for imags
        :param day_or_night: Use day (D), night (N) or Day & Night data (DNB)
        """
        assert len(center) == 2
        assert len(margin) == 2

        self.date = datetime(year, month, day)
        self.center = center
        self.margin = margin
        self.dayOrNight = day_or_night
        self.files = get_data(DATA_DIRECTORY, year, month, day, west=center[1] - margin[1], east=center[1] + margin[1],
                              north=center[0] + margin[0], south=center[0] - margin[0], dayOrNight=self.dayOrNight)

        self.raw_scene = Scene(filenames=self.files, reader="viirs_l1b")  # Raw Scene object storing the raw data
        self.raw_scene.load(["I04", "I05", "i_lat", "i_lon"])  # Load the two primary bands of inspection + geodata

        area = create_area_def("eye_area",
                               {"proj": "lcc", "ellps": "WGS84", "lat_0": self.center[0], "lon_0": self.center[1],
                                "lat_1": self.center[0]},
                               width=200, height=200, units="degrees",
                               area_extent=[self.center[1] - self.margin[1], self.center[0] - self.margin[0],
                                            self.center[1] + self.margin[1], self.center[0] + self.margin[0]]
                               )
        self.core_scene = self.raw_scene.resample(area)

    def plot_globe(self, band="I04", proj="lcc", sf=1.0):
        try:
            crs = self.corrected_scene[band].attrs["area"].to_cartopy_crs()
        except AttributeError:
            area = self.raw_scene[band].attrs["area"].compute_optimal_bb_area(
                {"proj": proj, "lat_0": self.center[0], "lon_0": self.center[1], "lat_1": 10, "lat_2": 20}
            )
            self.corrected_scene = self.raw_scene.resample(area)  # resample image to projection
            crs = self.corrected_scene[band].attrs["area"].to_cartopy_crs()
        ax = plt.axes(projection=crs)
        # Cartopy methods
        ax.coastlines()
        ax.gridlines()
        ax.set_global()
        plt.imshow(self.corrected_scene[band], transform=crs,
                   extent=(
                       crs.bounds[0] * sf, crs.bounds[1] * sf, crs.bounds[2] * sf,
                       crs.bounds[3] * sf),
                   origin="upper")
        cb = plt.colorbar()
        cb.set_label("Kelvin (K)")
        plt.show()

    def draw_rect(self, center, w, h):
        splice = self.core_scene.crop(xy_bbox=[center[0] - w/2,center[1] - h/2,center[0] + w/2,center[1] + h/2])

        plt.scatter(splice["I04"].data.flatten().compute(), splice["I05"].data.flatten().compute())
        plt.show()


if __name__ == "__main__":

    with ProgressBar():
        ci = CycloneImage(2017, 9, 19, center=(16.58, -63.52), margin=(0.5, 0.5))
        ci.draw_rect((-20000, 0), 5000, 40000)
