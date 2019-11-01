import matplotlib.pyplot as plt
import netCDF4 as nt
import numpy as np
from fetch_file import get_data
from CycloneImage import CycloneImage


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


def compare_diff_days(filenames, i05_temps, i04_temps, eyes, width, sides):
    for i in range(len(i04_temps)):
        if sides[i] == "left":
            i05eye = i05_temps[i][eyes[i][1] - 150:eyes[i][1], eyes[i][0] - width:eyes[i][0] + width]
            i04eye = i04_temps[i][eyes[i][1] - 150:eyes[i][1], eyes[i][0] - width:eyes[i][0] + width]
        elif sides[i] == "right":
            i05eye = i05_temps[i][eyes[i][1]:eyes[i][1]+150, eyes[i][0] - width:eyes[i][0] + width]
            i04eye = i04_temps[i][eyes[i][1]:eyes[i][1]+150, eyes[i][0] - width:eyes[i][0] + width]
        i05t = np.mean(i05eye, axis=1)
        i04t = np.mean(i04eye, axis=1)
        label = filenames[i][18:30]
        plt.scatter(i04t, i05t, label=label, s=20)
    plt.xlabel('I04')
    plt.ylabel('I05')
    plt.legend()


if __name__ == "__main__":
    # ci = CycloneImage(2017, 9, 19, center=(16, -62.5), margin=(2.5, 2.5))
    # ci.plot_globe()

    #  filenames = get_data(root_dir="../Data", year=2019, month=9, day=26, north=16, south=16, west=-41, east=-40, dayOrNight="D")
    # for filename in filenames:   # ['viirs_l1b']:
    #    if 'VNP02' in filename:
    #        print(filename)
    #        temps_i05 = load_file(filename)
    #        #temps_i04 = load_file(filename, band="I04")
    #        plot_whole_im(temps_i05)

    #same_storm_files = ["../Data/VNP02IMG.A2019242.1730.001.2019243030442.nc",
    #         "../Data/VNP02IMG.A2019243.1712.001.2019243231515.nc",
    #         "../Data/VNP02IMG.A2019244.1830.001.2019245003519.nc"]
    # same_storm_eyes = [[4245, 2413], [5916, 2166], [2103, 6217]]

    diff_storm_files = ["../Data/VNP02IMG.A2017262.1742.001.2017335035656.nc",
                        "../Data/VNP02IMG.A2018283.1848.001.2018284055739.nc",
                        "../Data/VNP02IMG.A2019242.1730.001.2019243030442.nc",
                        "../Data/VNP02IMG.A2019269.1536.001.2019276171936.nc"]
    diff_storm_eyes = [[2360, 330], [2995, 1891], [4245, 2413], [4147, 6160]]
    directions = ["left", "left", "left", "left"]
    i04_temp_list = []
    i05_temp_list = []
    for file in diff_storm_files:
        temps_i05 = load_file(file)
        temps_i04 = load_file(file, band="I04")
        i04_temp_list.append(temps_i04)
        i05_temp_list.append(temps_i05)
    #plot_eye(i05_temp_list[0], eye_y=4245, eye_x=2413)
    compare_diff_days(diff_storm_files, i05_temp_list, i04_temp_list, diff_storm_eyes, width=3, sides=directions)

    plt.show()
