from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pyresample import image, geometry
import numpy as np

path = "C:\\Users\\Robert\\Downloads\\new.002.c6.nc"

with Dataset(path) as rootel:
    snap = rootel["AODc_int"][0]
    lat, lon = np.meshgrid(rootel["lat"], rootel["lon"])
    swath = geometry.SwathDefinition(lats=lat, lons=lon)
    image = image.ImageContainerBilinear(snap, swath, radius_of_influence=10000)
    area = swath.compute_optimal_bb_area({"proj": "merc", "lat_ts": 0})
    crs = area.to_cartopy_crs()
    ax = plt.axes(projection=crs)
    ax.coastlines()
    ax.gridlines()
    ax.set_global()
    new_img = image.resample(area)
    ax.imshow(new_img.image_data, transform=crs, extent=(
        crs.bounds[0], crs.bounds[1], crs.bounds[2],
        crs.bounds[3]),
              origin="upper")
    plt.show()
