from datetime import datetime

import matplotlib.pyplot as plt
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

    def plot_globe(self, band="I04", proj="lcc"):
        area = self.raw_scene[band].attrs["area"].compute_optimal_bb_area(
            {"proj": proj, "lat_0": self.center[0], "lon_0": self.center[1], "lat_1": 25., "lat_2": 25.}
        )
        corrected_scene = self.raw_scene.resample(area)  # resample image to projection
        crs = corrected_scene[band].attrs["area"].to_cartopy_crs()

        # Cartopy methods
        ax = plt.axes(projection=crs)
        ax.coastlines()
        ax.gridlines()
        ax.set_global()

        plt.imshow(corrected_scene[band], transform=crs, extent=crs.bounds, origin="upper")
        cb = plt.colorbar()
        cb.set_label("Kelvin (K)")
        plt.show()


if __name__ == "__main__":
    pass
