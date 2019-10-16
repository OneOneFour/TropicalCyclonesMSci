import netCDF4 as nt
import matplotlib.pyplot as plt


rootgrp = nt.Dataset("data/temperature/NPPSoumi 2017-09-19/VNP02IMG.A2017262.1742.001.2017335035656.nc", "r+",
                     format="NETCDF5")
obs_dat = rootgrp.groups["observation_data"].variables
lat, long = rootgrp.GRingPointLatitude, rootgrp.GRingPointLongitude
obs_data_Io4, obs_data_Io5 = obs_dat['I04'], obs_dat['I05']


obs_io5_lut = obs_dat["I05_brightness_temperature_lut"]

obs_io5_temp.reshape(obs_data_Io5.shape)

fig, ax = plt.subplots()
im = ax.imshow(obs_io5_temp, cmap="jet", aspect="auto")
fig.colorbar(im)
plt.show()

rootgrp.close()