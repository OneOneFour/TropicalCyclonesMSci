import csv
import datetime

import matplotlib.pyplot as plt
import netCDF4 as nt
import numpy as np
from dask.diagnostics.progress import ProgressBar

from CycloneImage import CycloneImage
from fetch_file import get_data


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
            obs_data_band /= (
                    1 + obs_uncert * rootgrp.groups["observation_data"].variables[band].radiance_scale_factor ** 2)
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
    eye = temps[eye_y - padding:eye_y + padding, eye_x - padding: eye_x + padding]
    fig, ax = plt.subplots()
    im = ax.imshow(eye, cmap="jet")
    fig.colorbar(im)


def plot_whole_im(temps):
    fig, ax = plt.subplots()
    im = ax.imshow(temps, cmap="jet")
    fig.colorbar(im)


def rect_sample_profile(i05temps, i04temps, eye_x, eye_y, width=5, max_r=150, type='both'):
    i05eye = i05temps[eye_x:eye_x - max_r:-1, eye_y - width:eye_y + width]
    i04eye = i04temps[eye_x:eye_x - max_r:-1, eye_y - width:eye_y + width]
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


def compare_diff_days(filenames, i05_temps, i04_temps, eyes, width, sides, radius):
    num_days = len(filenames)
    half_days_floor = int(np.floor(num_days / 2))
    half_days_ceil = int(np.ceil(num_days / 2))
    fig, axs = plt.subplots(half_days_floor, half_days_ceil)
    subplot_y = subplot_x = 0
    for i in range(len(i04_temps)):
        if sides[i] == "left":
            i05eye = i05_temps[i][eyes[i][1] - width:eyes[i][1] + width, eyes[i][0] - radius:eyes[i][0]]
            i04eye = i04_temps[i][eyes[i][1] - width:eyes[i][1] + width, eyes[i][0] - radius:eyes[i][0]]
        elif sides[i] == "right":
            i05eye = i05_temps[i][eyes[i][1] - width:eyes[i][1] + width, eyes[i][0] - radius:eyes[i][0]]
            i04eye = i04_temps[i][eyes[i][1] - width:eyes[i][1] + width, eyes[i][0] - radius:eyes[i][0]]
        i05t = np.mean(i05eye, axis=0)
        i04t = np.mean(i04eye, axis=0)
        label = filenames[i][18:30]
        axs[subplot_y, subplot_x].scatter(i04t, i05t, label=label, s=20)
        axs[subplot_y, subplot_x].set_xlabel('I04')
        axs[subplot_y, subplot_x].set_ylabel('I05')
        axs[subplot_y, subplot_x].legend()
        if subplot_x == half_days_ceil - 1:
            subplot_x = 0
            subplot_y += 1
        else:
            subplot_x += 1


def load_cycs_and_plot(filenames, eyes, directions, plot_type="eye"):
    i04_temp_list = []
    i05_temp_list = []
    for file in filenames:
        temps_i05 = load_file(file)
        temps_i04 = load_file(file, band="I04")
        i04_temp_list.append(temps_i04)
        i05_temp_list.append(temps_i05)

    if plot_type == "eye":
        for i in range(len(i05_temp_list)):
            plot_eye(i05_temp_list[i], eye_x=eyes[i][0], eye_y=eyes[i][1], padding=75)
    elif plot_type == "prof_compare":
        compare_diff_days(filenames, i05_temp_list, i04_temp_list, eyes, width=3, radius=130, sides=directions)


def read_cyc_csv(filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        filenames = []
        eye_data = []
        eye_directions = []
        for row in csv_reader:
            filenames.append('Data/%s' % row['Filename'])
            eye_data.append([int(row['Eye X']), int(row['Eye Y'])])
            eye_directions.append(row['Direction'])
    return filenames, eye_data, eye_directions


def find_cyc_data(year, month, day, eye_lat, eye_long, where_store):
    filenames = get_data(root_dir="Data", year=year, month=month, day=day,
                         north=eye_lat + 1, south=eye_lat - 1, west=eye_long - 1, east=eye_long + 1,
                         dayOrNight="D")
    for filename in filenames:
        if 'VNP02' in filename:
            print(filename)
            temps_i05 = load_file(filename)
            plot_whole_im(temps_i05)
            plt.pause(15)
            eyex = input("What is the eye x?")
            eyey = input("What is the eye y?")
            side = "left" if int(eyex) <= 3200 else "right"
            with open(where_store, mode='a') as csv_file:
                csv_writer = csv.writer(csv_file)
                date = datetime.datetime(int(filename[15:19]), 1, 1) + datetime.timedelta(int(filename[19:22]) - 1)
                row = [filename[5:], filename[15:19], date.date().month, date.date().day, filename[23:27], eyex, eyey,
                       side, 0, 3]
                csv_writer.writerow(row)


def glob_pickle_files(directory):
    from glob import glob
    return glob(f"{directory}\*.pickle")


def pickle_file():
    fname = input("Enter file path of cyclone pickle")
    with ProgressBar():
        ci = CycloneImage.load_cyclone_image(fname)
        ci.draw_eye("I05")


if __name__ == "__main__":
    #dirpath = input("Enter directory containing pickle files")
    dirpath = "proc/pic_dat"
    with ProgressBar():
        pickle_paths = glob_pickle_files(dirpath)
        for pickle in pickle_paths:
            ci = CycloneImage.load_cyclone_image(pickle)
            filename_idx = 0
            if ci.is_complete:
                hottest_idx, l, r, t, b = ci.find_eye()
                eye_centre_idx = (r - l)/2 + l, (b - t)/2 + t
                for y in [-1, 1]:
                    for x in [-1, 1]:
                        filename_idx += 1
                        cyclone_centre_m_x = int(-ci.pixel_x * ci.I04.shape[0] * 0.5 + eye_centre_idx[0] * ci.pixel_x + ci.rmw * x/4)
                        cyclone_centre_m_y = int(ci.pixel_y * ci.I04.shape[1] * 0.5 - eye_centre_idx[1] * ci.pixel_y - ci.rmw * y/4)
                        x_pixel_centre = eye_centre_idx[0] + (ci.rmw/ci.pixel_x) *x/4
                        y_pixel_centre = eye_centre_idx[1] + (ci.rmw/ci.pixel_y) *y/4
                        try:
                            ci.draw_rect((cyclone_centre_m_x, cyclone_centre_m_y),  ci.rmw/2, ci.rmw/2, (x_pixel_centre, y_pixel_centre), filename_idx)
                        except IndexError:
                            print("Data outside image")

