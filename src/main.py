import netCDF4 as nt
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["PROJ_LIB"] = "C:\\Users\\tpklo\\.conda\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share"
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid


def basic_plot():
    rootgrp = nt.Dataset("C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2018283.1848.001.2018284055739.nc",
                          "r+", format="NETCDF5")
    obs_dat = rootgrp.groups["observation_data"].variables
    lat, long = rootgrp.GRingPointLatitude, rootgrp.GRingPointLongitude
    obs_data_Io4, obs_data_Io5 = obs_dat['I04'], obs_dat['I05']
    obs_io5_lut = obs_dat["I05_brightness_temperature_lut"]
    fig, ax = plt.subplots()
    im = ax.imshow(obs_data_Io5, cmap="jet", aspect="auto")
    fig.colorbar(im)
    plt.show()


def readin_filename(filename, names, scale=True):
    data = nt.Dataset(filename)
    data.set_auto_scale(False)
    indata = {}
    for name in names:
        groups = data.groups.keys()
        for group in groups:
            if name in data.groups[group].variables.keys():
                data_group = group
        indata[name] = data.groups[data_group].variables[name][:]
        indata[name] = indata[name].astype('float')
        if scale:
            indata[name][indata[name] > 65530] = np.nan
            try:
                indata[name] *= data.groups[data_group].variables[name].scale_factor
            except AttributeError:
                pass
            try:
                indata[name] += data.groups[data_group].variables[name].add_offset
            except AttributeError:
                pass
    return indata


def trunc_bright_temp(band, eye_coords, buffer):
    # Parameters
    filename = "C:/Users/tpklo/OneDrive/Documents/MSci/InitialCode/Data/VNP02IMG.A2017262.1742.001.2017335035656.nc"

    # Data Read In
    data = readin_filename(filename, [band, band+'_brightness_temperature_lut'], scale=False)
    bt = data[band+'_brightness_temperature_lut'][data[band].astype('int')]

    bt_trunc = np.empty([2*buffer, 2*buffer])
    bt_trunc_vert = bt[eye_coords[0]-buffer:eye_coords[0]+buffer]
    for i in range(len(bt_trunc_vert)):
        bt_trunc[i] = bt_trunc_vert[i][eye_coords[1]-buffer:eye_coords[1]+buffer]

    return bt_trunc


def basemap():
    map = Basemap(width=6000000, height=4500000, rsphere=(6378137.00, 6356752.3142), resolution='c',
                  area_thresh=1000., projection='lcc', lat_1=21., lat_2=46, lat_0=33.5, lon_0=-89.)
    map.drawcoastlines()
    map.drawmapboundary()
    plt.show()


def eye_plot(bands=['I04', 'I05'], eye_coords=[330, 2360], buffer=60):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for band, ax in zip(bands, axs):
        eye_data = trunc_bright_temp(band, eye_coords, buffer)
        im = ax.imshow(eye_data, cmap="jet", aspect="auto")
        ax.title.set_text(band)

    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

def profile(bands=['I04', 'I05'], eye_coords=[330, 2360]):


# vlims 2018 [1825, 1960]
# hlims 2018 [2920, 3050]

eye_plot()
