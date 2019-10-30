import matplotlib.pyplot as plt
import netCDF4 as nt
import numpy as np
from satpy import Scene

from fetch_file import get_data


def get_nc_files(year, month, day, time, ext=".nc"):
    from glob import glob
    import os
    str = os.path.join("..", "data", f"NPPSoumi {year}-{month}-{day}", f"*{ext}")
    return glob(str)


def load_file(img_file, geo_file=None, band="I05"):
    """Read in NETCDF4 file - return scaled map"""
    with nt.Dataset(img_file, "r+", format="NETCDF5") as rootgrp:
        try:
            obs_data_band = rootgrp.groups["observation_data"].variables[band][:]
            obs_lookup_band = rootgrp.groups["observation_data"].variables[band + "_brightness_temperature_lut"][:]
            obs_data_band[obs_data_band == rootgrp.groups["observation_data"].variables[band]._FillValue] = np.nan
            try:
                obs_data_band += rootgrp.groups["observation_data"].variables[band].add_offset
                obs_data_band /= rootgrp.groups["observation_data"].variables[band].scale_factor
            except AttributeError as e:
                raise e
        except KeyError as e:
            raise e
    if geo_file is not None:
        with nt.Dataset(geo_file, "r+", format="NETCDF5") as rootgrp:
            try:
                latitude = rootgrp.groups["geolocation_data"].variables["latitude"]
                longitude = rootgrp.groups["geolocation_data"].variables["longitude"]

                return obs_lookup_band[obs_data_band.astype("int")], latitude, longitude
            except KeyError as e:
                print("Are you sure this is a geolocation file")
                raise e

    return obs_lookup_band[obs_data_band.astype("int")]


def select_and_plot(temps, eye_x, eye_y, padding=50):
    eye = temps[eye_x - padding:eye_x + padding, eye_y - padding: eye_y + padding]
    fig, ax = plt.subplots()
    im = ax.imshow(eye, cmap="jet")
    fig.colorbar(im)
    plt.show()


def plot_using_imshow(temps):
    fig, ax = plt.subplots()
    im = ax.imshow(temps, cmap="jet")
    fig.colorbar(im)
    plt.show()


def rect_sample_profile(temps, eye_x, eye_y, width=5, max_r=150):
    eye = temps[eye_x:eye_x - max_r:-1, eye_y - width:eye_y + width]
    r = np.arange(0, max_r)
    t = np.mean(eye, axis=1)
    plt.plot(r, t)
    plt.show()


filenames = get_data("data", 2017, 9, 19, north=18, south=14, west=-65, east=-60,dayOrNight="D")

print(filenames)
scene = Scene(reader="viirs_l1b", filenames=filenames)
scene.load(["I04", "I05", "i_lat", "i_lon"])
new_scene = scene.resample("cwa")
plt.imshow(scene["I04"])
plt.colorbar()
plt.show()
plt.imshow(new_scene.resample("cwa"))
plt.colorbar()
plt.show()
# files = get_nc_files(2017, 9, 19)
# temps_i05 = load_file("../data/NPPSoumi 2017-9-19/VNP03IMG.A2017262.1600.001.2017335033838.nc")
# temps_i04 = load_file("../data/NPPSoumi 2017-9-19/VNP02IMG.A2017262.1742.001.2017335035656.nc", band="I04")
# t = temps_i05 - temps_i04
# select_and_plot(t, 330, 2360)
# rect_sample_profile(t, 330, 2360, max_r=75, width=10)
