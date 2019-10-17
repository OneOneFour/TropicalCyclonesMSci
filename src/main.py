import netCDF4 as nt
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.basemap import Basemap


def get_nc_files(year, month, day, ext=".nc"):
    from glob import glob
    import os
    str = os.path.join("..", "data", f"NPPSoumi {year}-{month}-{day}", f"*{ext}")
    return glob(str)


def load_file(file, band="I05"):
    """Read in NETCDF4 file - return scaled map"""
    with nt.Dataset(file, "r+", format="NETCDF5") as rootgrp:
        lat_anchors = rootgrp.GRingPointLatitude
        long_anchors = rootgrp.GRingPointLongitude
        try:
            obs_data_band = rootgrp.groups["observation_data"].variables[band][:]
            obs_lookup_band = rootgrp.groups["observation_data"].variables[band + "_brightness_temperature_lut"][:]
            obs_data_band[obs_data_band == rootgrp.groups["observation_data"].variables[band]._FillValue] = np.nan
            try:
                obs_data_band += rootgrp.groups["observation_data"].variables[band].add_offset
                obs_data_band /= rootgrp.groups["observation_data"].variables[band].scale_factor
            except AttributeError as e:
                raise e
            # lat = np.zeros(obs_data_band.shape)
            # long = np.zeros(obs_data_band.shape)
            #
            # lat[]
            # xv, yv = np.meshgrid(np.arange(0, obs_data_band.shape[0]), np.arange(0, obs_data_band.shape[1]))
            # long = long_anchors[0] + xv * (long_anchors[1] - long_anchors[0]) / ob + yv * (
            #             long_anchors[3] - long_anchors[0])
            # long =
            return obs_lookup_band[obs_data_band.astype("int")]
        except KeyError as e:
            raise e


def select_and_plot(temps, eye_x, eye_y, padding=50):
    eye = temps[eye_x - padding:eye_x + padding, eye_y - padding: eye_y + padding]
    fig, ax = plt.subplots()
    im = ax.imshow(eye, cmap="jet")
    fig.colorbar(im)
    plt.show()


def plot_using_bmap(temperatures, lat, longs):
    bmap = Basemap(width=100000, height=100000, resolution="l", projection="stere", lat_0=lat.mean(),
                   lon_0=longs.mean())
    la, lo = np.meshgrid(lat, longs)
    x_i, y_i = bmap(la, lo)

    bmap.pcolor(x_i, y_i, temperatures)
    bmap.drawcountries()
    plt.show()


def plot_using_imshow(temps):
    fig, ax = plt.subplots()
    im = ax.imshow(temps, cmap="jet")
    fig.colorbar(im)
    plt.show()


def rect_sample_profile(temps, eye_x, eye_y,width=5, max_r=150 ):
    eye = temps[eye_x:eye_x-max_r:-1, eye_y - width:eye_y + width]
    r = np.arange(0, max_r)
    t = np.mean(eye, axis=1)
    plt.plot(r,t)
    plt.show()

files = get_nc_files(2017, 9, 19)
temps_i05 = load_file("../data/NPPSoumi 2017-9-19/VNP02IMG.A2017262.1742.001.2017335035656.nc")
temps_i04 = load_file("../data/NPPSoumi 2017-9-19/VNP02IMG.A2017262.1742.001.2017335035656.nc", band="I04")
t = temps_i05 - temps_i04
select_and_plot(t, 330, 2360)
rect_sample_profile(t,330,2360,max_r = 75,width=10)
