import numpy as np
from netCDF4 import Dataset
import glob
import matplotlib.pyplot as plt


def find_trmm_profile(time_in, lat_in, lon_in):
    YYYY = time_in[0: 4]
    MM = time_in[5: 7]
    DD = time_in[8:10]
    HH = time_in[11:13]
    INPUT_LAT = lat_in
    INPUT_LON = lon_in
    RADIUS_OUT = 40
    rain_profile = np.nan * np.zeros(RADIUS_OUT + 1)
    TRMM_LAT = np.arange(-49.875, 49.875 + 1e-6, 0.25)
    TRMM_LON = np.arange(-179.875, 179.875 + 1e-6, 0.25)
    INDEX_LAT = np.argmin(abs(TRMM_LAT - INPUT_LAT))
    INDEX_LON = np.argmin(abs(TRMM_LON - INPUT_LON))
    # mask out the area larger than deined RADIUS_OUT
    X = np.linspace(0, RADIUS_OUT * 2, RADIUS_OUT * 2 + 1)
    INDEX_i, INDEX_j = np.meshgrid(X, X)
    rainmask = np.sqrt((INDEX_i - RADIUS_OUT) ** 2 + (INDEX_j - RADIUS_OUT) ** 2)
    rainmask[rainmask > RADIUS_OUT] = np.nan
    #
    trmm_file_name = sorted(
        glob.glob('/net/wrfstore6-10/disk1/sw1013/TRMM/3B42.' + str(YYYY) + str(MM) + str(DD) + '.' + str(HH) + '*.nc'))
    #
    print(YYYY)
    print(MM)
    print(DD)
    print(HH)
    if trmm_file_name != []:
        fh = Dataset(trmm_file_name[0])
        pcp = fh.variables['pcp'][0]
        # INDEX_i,INDEX_j = np.meshgrid(X,Y)
        if not ((INDEX_LAT - RADIUS_OUT) < 0 or INDEX_LAT + RADIUS_OUT + 1 > len(TRMM_LAT) or (
                INDEX_LON - RADIUS_OUT) < 0 or INDEX_LON + RADIUS_OUT + 1 > len(TRMM_LON)):
            tmp_rain_field = pcp[INDEX_LAT - RADIUS_OUT:INDEX_LAT + RADIUS_OUT + 1,
                             INDEX_LON - RADIUS_OUT:INDEX_LON + RADIUS_OUT + 1]
            tmp_rain_field[np.isnan(rainmask)] = np.nan
            # filter out the rainfield with missing values within 5degree radius
            if np.all(tmp_rain_field.data[~np.isnan(rainmask)] != -9999.9):
                for rrr in range(RADIUS_OUT + 1):
                    rain_profile[rrr] = np.nanmean(tmp_rain_field[(rainmask > rrr - 0.5) & (rainmask < rrr + 0.5)])
        # TC location close to day line, roll pcp
        if ((INDEX_LAT - RADIUS_OUT) >= 0 and INDEX_LAT + RADIUS_OUT + 1 <= len(TRMM_LAT)) and (
                (INDEX_LON - RADIUS_OUT) < 0 or INDEX_LON + RADIUS_OUT + 1 > len(TRMM_LON)):
            pcp = np.roll(pcp, 720, axis=1)
            TRMM_LON = np.roll(TRMM_LON, 720)
            INDEX_LON = np.argmin(abs(TRMM_LON - INPUT_LON))
            tmp_rain_field = pcp[INDEX_LAT - RADIUS_OUT:INDEX_LAT + RADIUS_OUT + 1,
                             INDEX_LON - RADIUS_OUT:INDEX_LON + RADIUS_OUT + 1]
            tmp_rain_field[np.isnan(rainmask)] = np.nan
            # filter out the rainfield with missing values within 5degree radius
            if np.all(tmp_rain_field.data[~np.isnan(rainmask)] != -9999.9):
                for rrr in range(RADIUS_OUT + 1):
                    rain_profile[rrr] = np.nanmean(tmp_rain_field[(rainmask > rrr - 0.5) & (rainmask < rrr + 0.5)])
    return (rain_profile)


if __name__ == "__main__":
    timein = "2018-03-10-12"
    lat = 18.8
    lon = -169.7
    profile = find_trmm_profile(timein, lat, lon)
    radius = np.arange(0, 10.25, 0.25)
    plt.plot(radius, profile)
    plt.savefig("./fig.png")
    print(profile)
