import netCDF4 as nt
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['PROJ_LIB'] = 'C:\\Users\\tpklo\\.conda\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share'

from satpy import Scene, find_files_and_readers
from datetime import datetime

from mpl_toolkits.basemap import Basemap

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



filenames = find_files_and_readers(base_dir="../data", reader="viirs_l1b", start_time=datetime(2017, 9, 19),
                                   end_time=datetime(2017, 9, 20))

print(filenames)
scene = Scene(reader="viirs_l1b",filenames=filenames)
scene.load(["I04","I05"])
scene.show("I04")
# files = get_nc_files(2017, 9, 19)
# temps_i05 = load_file("../data/NPPSoumi 2017-9-19/VNP03IMG.A2017262.1600.001.2017335033838.nc")
# temps_i04 = load_file("../data/NPPSoumi 2017-9-19/VNP02IMG.A2017262.1742.001.2017335035656.nc", band="I04")
# t = temps_i05 - temps_i04
# select_and_plot(t, 330, 2360)
# rect_sample_profile(t, 330, 2360, max_r=75, width=10)
files = get_nc_files(2017, 9, 19)
temps_i05 = load_file("C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2017262.1742.001.2017335035656.nc")
temps_i04 = load_file("C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2017262.1742.001.2017335035656.nc", band="I04")
t = temps_i05 - temps_i04
select_and_plot(t, 330, 2360)
rect_sample_profile(temps_i05, temps_i04, 330, 2360, max_r=75, width=10, type='both')

plt.show()
