from datetime import datetime

import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from datetime import timedelta
from pyresample import create_area_def
from satpy import Scene
import pickle
from fetch_file import get_data, get_data_single_date

DATA_DIRECTORY = "data"
DEFAULT_MARGIN = 0.5


def nm_to_degrees(nm):
    return nm / 60


def get_eye(start_point, end_point, **kwargs):
    lat = start_point["LAT"], end_point["LAT"]
    lon = start_point["LON"], end_point["LON"]
    radMaxWind = (start_point["USA_RMW"] + end_point["USA_RMW"]) / 30
    files = get_data(DATA_DIRECTORY, start_point["ISO_TIME"].to_pydatetime(), end_point["ISO_TIME"].to_pydatetime(),
                     north=max(lat) + DEFAULT_MARGIN,
                     south=min(lat) - DEFAULT_MARGIN, east=max(lon) + DEFAULT_MARGIN, west=min(lon) - DEFAULT_MARGIN)
    if files is None:
        print("No files were found")
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
                           width=200, height=200, units="degrees",
                           area_extent=[lon_int - radMaxWind, lat_int - radMaxWind,
                                        lon_int + radMaxWind, lat_int + radMaxWind]
                           )
    core_scene = raw_scene.resample(area)
    return CycloneImage(core_scene, center=(lat_int, lon_int), margin=radMaxWind, **kwargs)


class CycloneImage:

    @staticmethod
    def load_cyclone_image(fpath):
        with open(fpath, "rb") as file:
            ci = pickle.load(file)
        assert isinstance(ci, CycloneImage)
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
        else:
            raise ValueError("You must provide either a Scene object or a filepath to a scene object")
        self.center = center
        self.margin = DEFAULT_MARGIN
        self.day_or_night = "DNB"
        self.name = "UNKNOWN"
        for key, val in kwargs.items():
            self.__dict__[key] = val

    def save_object(self):
        file_name = f"{DATA_DIRECTORY}/proc/CORE_{self.name}_{self.core_scene.start_time.strftime('%Y_%m_%d')}.pickle"
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
        plt.figure()
        plt.imshow(self.core_scene[band])
        plt.title(f"{self.name} on {self.core_scene.start_time.strftime('%Y-%m-%d %H:%M:%S')} (Cat {self.cat})")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        cb = plt.colorbar()
        cb.set_label(f"{band} brightness temperature")
        plt.show()

    def draw_rect(self, center, w, h):
        plt.subplot(1, 2, 1)
        splice = self.core_scene.crop(
            xy_bbox=[center[0] - w / 2, center[1] - h / 2, center[0] + w / 2, center[1] + h / 2])

        plt.scatter(splice["I04"].data.flatten().compute(), splice["I05"].data.flatten().compute())
        plt.ylabel("Cloud Top Temperature (K)")
        plt.xlabel("I4 band reflectance (K)")
        plt.subplot(1, 2, 2)
        plt.imshow(self.core_scene["I04"])
        plt.show()


if __name__ == "__main__":
    with ProgressBar():
        ci = CycloneImage(2017, 9, 19, center=(16.58, -63.52), margin=(0.5, 0.5))
        ci.draw_rect((0, -30000), 5000, 40000)
