import netCDF4 as nt
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['PROJ_LIB'] = 'C:\\Users\\tpklo\\.conda\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share'


from mpl_toolkits.basemap import Basemap


def get_nc_files(year, month, day, ext=".nc"):
    from glob import glob
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


def plot_using_bmap(temperatures, lat, longs):
    bmap = Basemap(width=100000, height=100000, resolution="l", projection="stere", lat_0=lat.mean(),
                   lon_0=longs.mean())
    la, lo = np.meshgrid(lat, longs)
    x_i, y_i = bmap(la, lo)

    bmap.pcolor(x_i, y_i, temperatures)
    bmap.drawcountries()


def plot_using_imshow(temps):
    fig, ax = plt.subplots()
    im = ax.imshow(temps, cmap="jet")
    fig.colorbar(im)


def rect_sample_profile(i05temps, i04temps, eye_x, eye_y,width=5, max_r=150, type='both'):
    i05eye = i05temps[eye_x:eye_x-max_r:-1, eye_y - width:eye_y + width]
    i04eye = i04temps[eye_x:eye_x-max_r:-1, eye_y - width:eye_y + width]
    r = np.arange(0, max_r) * 375
    i05t = np.mean(i05eye, axis=1)
    i04t = np.mean(i04eye, axis=1)
    plt.figure()
    if type == 'both':
        plt.plot(r, i05t, label='I05')
        plt.plot(r, i04t, label='I04')
        plt.ylabel('I05/I04 Brightness Temperature/K')
        plt.xlabel('Radius from centre/m')
        plt.legend()
    elif type == 'compare':
        plt.plot(i05t-i04t, i05t)
        plt.xlabel('I05-I04')
        plt.ylabel('I05')


files = get_nc_files(2017, 9, 19)
temps_i05 = load_file("C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2017262.1742.001.2017335035656.nc")
temps_i04 = load_file("C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2017262.1742.001.2017335035656.nc", band="I04")
t = temps_i05 - temps_i04
select_and_plot(t, 330, 2360)
rect_sample_profile(temps_i05, temps_i04, 330, 2360, max_r=75, width=10, type='both')

plt.show()
