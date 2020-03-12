from fetch_file import get_data
import os
from satpy import Scene

DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", './')


class CycloneImageFast:
    """
    I/O class to read and dump files to a flattened and cropped image in .nc format
    Faster than existing data
    """
    BASE_DATASETS = ["I05", "I04"]
    OPTIONAL_DATASETS = ["M09", "I01"]

    @staticmethod
    def from_cyclone_image(ci):
        pass

    @classmethod
    def from_points(cls, start_point, end_point):
        lat_0 = (start_point["USA_LAT"] + end_point["USA_LAT"]) / 2
        lon_0 = (start_point["USA_LON"] + end_point["USA_LON"]) / 2
        north_extent = (start_point["USA_R34_NE"] + start_point["USA_R34_NW"]) / 120
        south_extent = (start_point["USA_R34_SE"] + start_point["USA_R34_SW"]) / 120
        west_extent = (start_point["USA_R34_SW"] + start_point["USA_R34_NW"]) / 120
        east_extent = (start_point["USA_R34_NE"] + start_point["USA_R34_NW"]) / 120

        try:
            files, urls = get_data(DATA_DIRECTORY, start_point["ISO_TIME"].to_pydatetime(),
                                   end_point["ISO_TIME"].to_pydatetime(),
                                   north=lat_0 + north_extent,
                                   south=lat_0 - south_extent,
                                   west=lon_0 - west_extent,
                                   east=lon_0 + east_extent,
                                   dayOrNight="D", include_mod=False)
        except FileNotFoundError:
            return None
        inst__ = cls(files)

    __slots__ = ["scene", "metadata"]

    def __init__(self, files):
        self.scene = Scene(filenames=files, reader="viirs_l1b")
        self.metadata = {}

    def __interpolate(self, start: dict, end: dict):
        from pandas import isna
        t = self.scene.start_time - start["ISO_TIME"].topydatetime()
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

    def write(self):
        self.scene.save_dataset(writer="cf", datasets=[])
