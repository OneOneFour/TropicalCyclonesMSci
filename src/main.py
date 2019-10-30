import netCDF4 as nt
from satpy import Scene, find_files_and_readers
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import h5py
#import os
#os.environ['PROJ_LIB'] = 'C:\\Users\\tpklo\\.conda\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share'


def get_nc_files(year, month, day, ext=".nc"):
    from glob import glob
    import os
    str = os.path.join("..", "data", f"NPPSoumi {year}-{month}-{day}", f"*{ext}")
    return glob(str)


def load_file(img_file, geo_file=None, band="I05"):
    """Read in NETCDF4 file - return scaled map"""
    with nt.Dataset(img_file, "r+", format="NETCDF5") as rootgrp:
        if band == "I04" or band == "I05":
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
        else:
            obs_data_band = rootgrp.groups["observation_data"].variables[band][:]
            obs_data_band += rootgrp.groups["observation_data"].variables[band].radiance_add_offset
            obs_uncert = rootgrp.groups["observation_data"].variables[band + "_uncert_index"][:]
            obs_data_band /= (1 + obs_uncert * rootgrp.groups["observation_data"].variables[band].radiance_scale_factor**2)
            return obs_data_band
    if geo_file is not None:
        with nt.Dataset(geo_file, "r+", format="NETCDF5") as rootgrp:
            try:
                latitude = rootgrp.groups["geolocation_data"].variables["latitude"][:]
                longitude = rootgrp.groups["geolocation_data"].variables["longitude"][:]
                return obs_lookup_band[obs_data_band.astype("int")], longitude, latitude
            except KeyError as e:
                print("Are you sure this is a geolocation file")
                raise e
    return obs_lookup_band[obs_data_band.astype("int")]


def plot_eye(temps, eye_x, eye_y, padding=50):
    eye = temps[eye_x - padding:eye_x + padding, eye_y - padding: eye_y + padding]
    fig, ax = plt.subplots()
    im = ax.imshow(eye, cmap="jet")
    fig.colorbar(im)


def plot_whole_im(temps):
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
        plt.plot(i04t, i05t)
        plt.xlabel('I04')
        plt.ylabel('I05')


def combined_imaging_bands(filename, eye_coords=None):
    bands = ["I01", "I02", "I03"]
    for band in bands:
        temps = load_file(
            "C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2017262.1742.001.2017335035656.nc",
            band=band)
        temps_arr = np.array(temps)
        if band == "I01":
            temps_combined = temps_arr
        else:
            temps_combined += temps_arr
    if eye_coords is None:
        plot_whole_im(temps_combined)
    else:
        plot_eye(temps_combined, eye_coords[0], eye_coords[1])

#filenames = find_files_and_readers(base_dir="../Data", reader="viirs_l1b", start_time=datetime(2017, 9, 19), end_time=datetime(2017, 9, 20))
#scene = Scene(reader="viirs_l1b", filenames=filenames)
#scene.load(["I04","I05"])
#files = get_nc_files(2017, 9, 19)

temps_i05, longs, lats = load_file("C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2017262.1742.001.2017335035656.nc",
                      geo_file="C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP03IMG.A2017262.1742.001.2017335033241.nc")
temps_i04 = load_file("C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2017262.1742.001.2017335035656.nc", band="I04")
rect_sample_profile(temps_i05, temps_i04, 330, 2360, max_r=75, width=5, type='compare')

plt.show()
