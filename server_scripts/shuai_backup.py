import warnings

warnings.filterwarnings("ignore")
from netCDF4 import Dataset
from matplotlib import gridspec
from scipy.optimize import curve_fit
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
from termcolor import cprint
from vincenty import vincenty
from matplotlib.colorbar import Colorbar
import pandas                  as pd
import numpy                   as np
import numpy                   as  np
import matplotlib.pyplot       as  plt
import pandas                  as  pd
import statsmodels.formula.api as  sm
import matplotlib
import scipy, scipy.stats, glob, copy

# define the year period
# TRMM  is from 1997 to 2014
# IMERG is from 2015 onwards
START_YEAR = 2004
END_YEAR = 2017
INDEX_year = np.arange(START_YEAR, END_YEAR + 1)
# basin names
BASIN_NAMEs = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP', 'SA', 'NH', 'SH', 'GB']
# Creat CAT_KT that contains the upper and lower winds for different categories
tmp = np.zeros((2, 7))
tmp[0, 0] = -1000.
tmp[1, 0] = 33.
tmp[0, 1] = 34.
tmp[1, 1] = 63.
tmp[0, 2] = 64.
tmp[1, 2] = 82.
tmp[0, 3] = 83.
tmp[1, 3] = 95.
tmp[0, 4] = 96.
tmp[1, 4] = 112.
tmp[0, 5] = 113.
tmp[1, 5] = 136.
tmp[0, 6] = 137.
tmp[1, 6] = 10000.
CAT_KT = pd.DataFrame(tmp, index=['LOWER', 'UPPER'], columns=['TD', 'TS', 'CAT1', 'CAT2', 'CAT3', 'CAT4', 'CAT5'])
cprint(CAT_KT, 'cyan', 'on_grey')
cprint('To call, use   e.g.  CAT_KT.loc[(\'UPPER\'),(\'CAT1\')]', 'cyan', 'on_grey')
# define the varibale names of lat lon wind pressure and size
AGENCY_LAT = ['USA_LAT', 'USA_LAT', 'USA_LAT', 'USA_LAT', 'USA_LAT', 'USA_LAT', 'USA_LAT']
AGENCY_LON = ['USA_LON', 'USA_LON', 'USA_LON', 'USA_LON', 'USA_LON', 'USA_LON', 'USA_LON']
AGENCY_MAX_WIND = ['USA_WIND', 'USA_WIND', 'USA_WIND', 'USA_WIND', 'USA_WIND', 'USA_WIND', 'USA_WIND']
AGENCY_MIN_PRES = ['USA_PRES', 'USA_PRES', 'USA_PRES', 'USA_PRES', 'USA_PRES', 'USA_PRES', 'USA_PRES']
AGENCY_RMW = ['USA_RMW', 'USA_RMW', 'USA_RMW', 'USA_RMW', 'USA_RMW', 'USA_RMW', 'USA_RMW']
AGENCY_R18_NE = ['USA_R34_NE', 'USA_R34_NE', 'USA_R34_NE', 'USA_R34_NE', 'USA_R34_NE', 'USA_R34_NE', 'USA_R34_NE']
AGENCY_R18_SE = ['USA_R34_SE', 'USA_R34_SE', 'USA_R34_SE', 'USA_R34_SE', 'USA_R34_SE', 'USA_R34_SE', 'USA_R34_SE']
AGENCY_R18_SW = ['USA_R34_SW', 'USA_R34_SW', 'USA_R34_SW', 'USA_R34_SW', 'USA_R34_SW', 'USA_R34_SW', 'USA_R34_SW']
AGENCY_R18_NW = ['USA_R34_NW', 'USA_R34_NW', 'USA_R34_NW', 'USA_R34_NW', 'USA_R34_NW', 'USA_R34_NW', 'USA_R34_NW']
AGENCY_R26_NE = ['USA_R50_NE', 'USA_R50_NE', 'USA_R50_NE', 'USA_R50_NE', 'USA_R50_NE', 'USA_R50_NE', 'USA_R50_NE']
AGENCY_R26_SE = ['USA_R50_SE', 'USA_R50_SE', 'USA_R50_SE', 'USA_R50_SE', 'USA_R50_SE', 'USA_R50_SE', 'USA_R50_SE']
AGENCY_R26_SW = ['USA_R50_SW', 'USA_R50_SW', 'USA_R50_SW', 'USA_R50_SW', 'USA_R50_SW', 'USA_R50_SW', 'USA_R50_SW']
AGENCY_R26_NW = ['USA_R50_NW', 'USA_R50_NW', 'USA_R50_NW', 'USA_R50_NW', 'USA_R50_NW', 'USA_R50_NW', 'USA_R50_NW']
AGENCY_R33_NE = ['USA_R64_NE', 'USA_R64_NE', 'USA_R64_NE', 'USA_R64_NE', 'USA_R64_NE', 'USA_R64_NE', 'USA_R64_NE']
AGENCY_R33_SE = ['USA_R64_SE', 'USA_R64_SE', 'USA_R64_SE', 'USA_R64_SE', 'USA_R64_SE', 'USA_R64_SE', 'USA_R64_SE']
AGENCY_R33_SW = ['USA_R64_SW', 'USA_R64_SW', 'USA_R64_SW', 'USA_R64_SW', 'USA_R64_SW', 'USA_R64_SW', 'USA_R64_SW']
AGENCY_R33_NW = ['USA_R64_NW', 'USA_R64_NW', 'USA_R64_NW', 'USA_R64_NW', 'USA_R64_NW', 'USA_R64_NW', 'USA_R64_NW']
# read in the whole ibtracs as a dataframe
ibt = pd.read_csv('/net/wrfstore5-10/disk1/sw1013/IBTrACS/ibtracs.ALL.list.v04r00.csv', skiprows=[1],
                  keep_default_na=False, na_values=[' '])
#
ETOPO_ncfile = '/net/wrfstore5-10/disk1/sw1013/ETOPO2v2/ETOPO2v2g_f4.nc'
fh = Dataset(ETOPO_ncfile)
ETOPO = fh.variables["z"][:].data
GRIDS_LON = fh.variables["x"][:].data
GRIDS_LAT = fh.variables["y"][:].data
INDEX_LON, INDEX_LAT = np.meshgrid(GRIDS_LON, GRIDS_LAT)
ETOPO = pd.DataFrame(ETOPO, index=GRIDS_LAT, columns=GRIDS_LON)


def land_ocean(ETOPO, lat_in, lon_in):
    # Creat ETOPO that contains the land depth in a unit of meter
    # To call, use ETOPO.loc[LAT,LON], LAT and LON should be float
    # < 0 is water, and >=0 is land
    lat_ind = np.argmin(np.abs(GRIDS_LAT - lat_in))
    lon_ind = np.argmin(np.abs(GRIDS_LON - lon_in))
    land_ocean = ETOPO.loc[GRIDS_LAT[lat_ind], GRIDS_LON[lon_ind]]
    return land_ocean


def utc2lst(utc, lon):
    # utc format %Y-%m-%d %H:%M:%S
    delta_sec = round(lon / 15) * 3600.
    time0 = datetime.strptime(utc, "%Y-%m-%d %H:%M:%S")
    lst = datetime.strftime(time0 + timedelta(0, delta_sec), "%Y-%m-%d %H:%M:%S")
    return lst


# define function to find rain volum
def find_trmm_rainvol(time_in, lat_in, lon_in, rsize_in):
    rainvol = np.nan
    rainvol_500 = np.nan
    rainvol_R34 = np.nan
    YYYY = time_in[0: 4]
    MM = time_in[5: 7]
    DD = time_in[8:10]
    HH = time_in[11:13]
    INPUT_LAT = lat_in
    INPUT_LON = lon_in
    RADIUS_OUT = 35
    TRMM_LAT = np.arange(-49.875, 49.875 + 1e-6, 0.25)
    TRMM_LON = np.arange(-179.875, 179.875 + 1e-6, 0.25)
    INDEX_LAT = np.argmin(abs(TRMM_LAT - INPUT_LAT))
    INDEX_LON = np.argmin(abs(TRMM_LON - INPUT_LON))
    # mask out the area larger than defined RADIUS_OUT
    X = np.linspace(0, RADIUS_OUT * 2, RADIUS_OUT * 2 + 1)
    INDEX_i, INDEX_j = np.meshgrid(X, X)
    rainmask_all = np.sqrt((INDEX_i - RADIUS_OUT) ** 2 + (INDEX_j - RADIUS_OUT) ** 2)
    dis_per_grid_km = ((vincenty([TRMM_LAT[INDEX_LAT] - 0.25 / 2, TRMM_LON[INDEX_LON]],
                                 [TRMM_LAT[INDEX_LAT] + 0.25 / 2, TRMM_LON[INDEX_LON]])) \
                       + vincenty([TRMM_LAT[INDEX_LAT], TRMM_LON[INDEX_LON] - 0.25 / 2],
                                  [TRMM_LAT[INDEX_LAT], TRMM_LON[INDEX_LON] + 0.25 / 2])) / 2.
    RADIUS_OUT = round((rsize_in * 1.852 * 1.2) / dis_per_grid_km)
    RADIUS_IN = round((rsize_in * 1.852 * 0.8) / dis_per_grid_km)
    rainmask = copy.copy(rainmask_all)
    rainmask[(rainmask_all > RADIUS_OUT) | (rainmask_all < RADIUS_IN)] = np.nan
    # rain vol with R34
    RADIUS_OUT = round((rsize_in * 1.852) / dis_per_grid_km)
    rainmask_R34 = copy.copy(rainmask_all)
    rainmask_R34[(rainmask_all > RADIUS_OUT)] = np.nan
    # rain vol with 500 km
    RADIUS_OUT = round((500) / dis_per_grid_km)
    rainmask_500 = copy.copy(rainmask_all)
    rainmask_500[(rainmask_all > RADIUS_OUT)] = np.nan
    ### debuging
    ###print(rsize_in*1.852,dis_per_grid_km,RADIUS_OUT,RADIUS_IN)
    ###rainmask [(rainmask>RADIUS_OUT) | (rainmask<RADIUS_IN)] = np.nan
    ###fig, ax = plt.subplots()
    ###ax.contourf(rainmask)
    ###circle1 = plt.Circle((20,20),rsize_in*1.852/dis_per_grid_km, color='m', fill=False, linewidth=1, linestyle='dashed')
    ###ax.add_artist(circle1)
    ###plt.show()
    #
    trmm_file_name = sorted(
        glob.glob('/net/wrfstore6-10/disk1/sw1013/TRMM/3B42.' + str(YYYY) + str(MM) + str(DD) + '.' + str(HH) + '*.nc'))
    #
    if trmm_file_name != []:
        fh = Dataset(trmm_file_name[0])
        pcp = fh.variables['pcp'][0]
        # INDEX_i,INDEX_j = np.meshgrid(X,Y)
        # rain vol within 500 km
        RADIUS_OUT = 35
        RADIUS_IN = -1
        if not ((INDEX_LAT - RADIUS_OUT) < 0 or INDEX_LAT + RADIUS_OUT + 1 > len(TRMM_LAT) or (
                INDEX_LON - RADIUS_OUT) < 0 or INDEX_LON + RADIUS_OUT + 1 > len(TRMM_LON)):
            tmp_rain_field = pcp[INDEX_LAT - RADIUS_OUT:INDEX_LAT + RADIUS_OUT + 1,
                             INDEX_LON - RADIUS_OUT:INDEX_LON + RADIUS_OUT + 1]
            tmp_rain_field[np.isnan(rainmask)] = np.nan
            # filter out the rainfield with missing values within 5degree radius
            if np.all(tmp_rain_field.data[~np.isnan(rainmask)] != -9999.9):
                rainvol = np.nansum(tmp_rain_field * 1e-3 * \
                                    (vincenty([TRMM_LAT[INDEX_LAT] - 0.25 / 2, TRMM_LON[INDEX_LON]],
                                              [TRMM_LAT[INDEX_LAT] + 0.25 / 2, TRMM_LON[INDEX_LON]]) * 1e3) \
                                    * vincenty([TRMM_LAT[INDEX_LAT], TRMM_LON[INDEX_LON] - 0.25 / 2],
                                               [TRMM_LAT[INDEX_LAT], TRMM_LON[INDEX_LON] + 0.25 / 2]) * 1e3)
        # rain vol within R34
        RADIUS_OUT = 35
        RADIUS_IN = -1
        if not ((INDEX_LAT - RADIUS_OUT) < 0 or INDEX_LAT + RADIUS_OUT + 1 > len(TRMM_LAT) or (
                INDEX_LON - RADIUS_OUT) < 0 or INDEX_LON + RADIUS_OUT + 1 > len(TRMM_LON)):
            tmp_rain_field = pcp[INDEX_LAT - RADIUS_OUT:INDEX_LAT + RADIUS_OUT + 1,
                             INDEX_LON - RADIUS_OUT:INDEX_LON + RADIUS_OUT + 1]
            tmp_rain_field[np.isnan(rainmask_R34)] = np.nan
            # filter out the rainfield with missing values within 5degree radius
            if np.all(tmp_rain_field.data[~np.isnan(rainmask)] != -9999.9):
                rainvol_R34 = np.nansum(tmp_rain_field * 1e-3 * \
                                        (vincenty([TRMM_LAT[INDEX_LAT] - 0.25 / 2, TRMM_LON[INDEX_LON]],
                                                  [TRMM_LAT[INDEX_LAT] + 0.25 / 2, TRMM_LON[INDEX_LON]]) * 1e3) \
                                        * vincenty([TRMM_LAT[INDEX_LAT], TRMM_LON[INDEX_LON] - 0.25 / 2],
                                                   [TRMM_LAT[INDEX_LAT], TRMM_LON[INDEX_LON] + 0.25 / 2]) * 1e3)
        # rain vol within 500 km
        RADIUS_OUT = 35
        RADIUS_IN = -1
        if not ((INDEX_LAT - RADIUS_OUT) < 0 or INDEX_LAT + RADIUS_OUT + 1 > len(TRMM_LAT) or (
                INDEX_LON - RADIUS_OUT) < 0 or INDEX_LON + RADIUS_OUT + 1 > len(TRMM_LON)):
            tmp_rain_field = pcp[INDEX_LAT - RADIUS_OUT:INDEX_LAT + RADIUS_OUT + 1,
                             INDEX_LON - RADIUS_OUT:INDEX_LON + RADIUS_OUT + 1]
            tmp_rain_field[np.isnan(rainmask_500)] = np.nan
            # filter out the rainfield with missing values within 5degree radius
            if np.all(tmp_rain_field.data[~np.isnan(rainmask)] != -9999.9):
                rainvol_500 = np.nansum(tmp_rain_field * 1e-3 * \
                                        (vincenty([TRMM_LAT[INDEX_LAT] - 0.25 / 2, TRMM_LON[INDEX_LON]],
                                                  [TRMM_LAT[INDEX_LAT] + 0.25 / 2, TRMM_LON[INDEX_LON]]) * 1e3) \
                                        * vincenty([TRMM_LAT[INDEX_LAT], TRMM_LON[INDEX_LON] - 0.25 / 2],
                                                   [TRMM_LAT[INDEX_LAT], TRMM_LON[INDEX_LON] + 0.25 / 2]) * 1e3)
    return (rainvol, rainvol_R34, rainvol_500)
    #


def find_trmm_profile(time_in, lat_in, lon_in, rsize_in):
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
    #


def find_hursat_btemp(time_in, lat_in, lon_in, rsize_in, sn_in):
    YYYY = time_in[0: 4]
    MM = time_in[5: 7]
    DD = time_in[8:10]
    HH = time_in[11:13]
    SIDE_GRIDs = 301
    PROFILE_GRIDs = 151
    GRID_KM = 8  # roughly 8 km grid spacing of HURSAT
    #
    btemp = np.nan
    btemp_profile = np.nan * np.zeros(PROFILE_GRIDs)
    #
    X = np.linspace(0, SIDE_GRIDs - 1, SIDE_GRIDs)
    INDEX_i, INDEX_j = np.meshgrid(X, X)
    hursatmask_km = np.sqrt((INDEX_i - (SIDE_GRIDs - 1) / 2.) ** 2 + (INDEX_j - (SIDE_GRIDs - 1) / 2.) ** 2) * GRID_KM
    hursatmask_grid = np.sqrt((INDEX_i - (SIDE_GRIDs - 1) / 2.) ** 2 + (INDEX_j - (SIDE_GRIDs - 1) / 2.) ** 2)
    RADIUS_OUT = rsize_in * 1.852 + GRID_KM * 3
    RADIUS_IN = rsize_in * 1.852 - GRID_KM * 3
    hursatmask_km[(hursatmask_km > RADIUS_OUT) | (hursatmask_km < RADIUS_IN)] = np.nan
    #
    trmm_file_name = sorted(glob.glob(
        '/net/wrfstore5-10/disk1/sw1013/HURSAT_V6_IR/' + sn_in[:4] + '/' + sn_in + '.*.' + str(YYYY) + '.' + str(
            MM) + '.' + str(DD) + '.' + str(HH) + '00.*.nc'))
    #
    tmp_btemp = []
    tmp_btemp_profile = []
    if trmm_file_name != []:
        for BFILE in range(len(trmm_file_name)):
            fh = Dataset(trmm_file_name[BFILE])
            irwin_tmp = fh.variables['IRWIN'][0]
            irwin_debug = copy.copy(irwin_tmp)
            irwin_tmp[irwin_tmp < 0] = np.nan
            # get the BTEMP around R34
            irwin = copy.copy(irwin_tmp)
            irwin[np.isnan(hursatmask_km)] = np.nan
            tmp_btemp.extend([np.nanmean(irwin)])
            # get the BTEMP profile
            tmp_btemp_profile.append(np.nan * np.zeros(PROFILE_GRIDs))
            for rrr in range(PROFILE_GRIDs):
                tmp_btemp_profile[-1][rrr] = np.nanmean(
                    irwin_tmp[(hursatmask_grid > rrr - 0.5) & (hursatmask_grid < rrr + 0.5)])
        btemp = np.nanmean(tmp_btemp)
        btemp_profile = np.nanmean(tmp_btemp_profile, 0)
    if np.isnan(btemp):
        print(time_in, lat_in, lon_in, rsize_in, sn_in)
    return (btemp, btemp_profile)
    #


def find_sn(data_in, SOURCE_LAT, SOURCE_LON, SOURCE_MAX_WIND, SOURCE_MIN_PRES, SOURCE_RSIZE_18_1, SOURCE_RSIZE_18_2,
            SOURCE_RSIZE_18_3, SOURCE_RSIZE_18_4, \
            SOURCE_RSIZE_26_1, SOURCE_RSIZE_26_2, SOURCE_RSIZE_26_3, SOURCE_RSIZE_26_4, \
            SOURCE_RSIZE_33_1, SOURCE_RSIZE_33_2, SOURCE_RSIZE_33_3, SOURCE_RSIZE_33_4):
    lat = np.asarray(data_in[SOURCE_LAT].values)
    lon = np.asarray(data_in[SOURCE_LON].values)
    vmax = np.asarray(data_in[SOURCE_MAX_WIND].values)
    pmin = np.asarray(data_in[SOURCE_MIN_PRES].values)
    # get size of interest (not for reference)
    tmp1 = np.asarray(data_in[SOURCE_RSIZE_18_1].values)
    tmp2 = np.asarray(data_in[SOURCE_RSIZE_18_2].values)
    tmp3 = np.asarray(data_in[SOURCE_RSIZE_18_3].values)
    tmp4 = np.asarray(data_in[SOURCE_RSIZE_18_4].values)
    tmp1[tmp1 <= 0] = np.nan
    tmp2[tmp2 <= 0] = np.nan
    tmp3[tmp3 <= 0] = np.nan
    tmp4[tmp4 <= 0] = np.nan
    rsize = np.nanmean([tmp1, tmp2, tmp3, tmp4], 0)
    # rsize      = (tmp1 + tmp2 + tmp3 + tmp4) / 4
    time = data_in['ISO_TIME'].values
    nature = data_in['NATURE'].values
    sn = data_in['SID'].values
    year = data_in['SEASON'].values
    name = data_in['NAME'].values
    sn_out = np.nan
    year_out = np.nan
    name_out = np.nan
    lat_out = np.asarray([np.nan])
    lon_out = np.asarray([np.nan])
    lat_used = np.asarray([np.nan])
    lon_used = np.asarray([np.nan])
    # remove inregular record not on 00 06 12 18
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][11:13]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lat = lat[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lon = lon[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    vmax = pmin[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    pmin = pmin[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    rsize = rsize[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    nature = nature[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    sn = sn[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    year = year[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    name = name[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    #
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][14:16]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00')]
    lat = lat[(tmp == '00')]
    lon = lon[(tmp == '00')]
    vmax = pmin[(tmp == '00')]
    pmin = pmin[(tmp == '00')]
    rsize = rsize[(tmp == '00')]
    nature = nature[(tmp == '00')]
    sn = sn[(tmp == '00')]
    year = year[(tmp == '00')]
    name = name[(tmp == '00')]
    #
    lat_withland = copy.copy(lat)
    lon_withland = copy.copy(lon)
    # TC nature control, remove all extratropical period
    time = time[nature != 'ET']
    lat = lat[nature != 'ET']
    lon = lon[nature != 'ET']
    vmax = vmax[nature != 'ET']
    pmin = pmin[nature != 'ET']
    rsize = rsize[nature != 'ET']
    sn = sn[nature != 'ET']
    year = year[nature != 'ET']
    name = name[nature != 'ET']
    nature = nature[nature != 'ET']
    # remove size over land
    for RRR in range(1, len(time) - 1):
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            if not (land_ocean(ETOPO, lat[RRR - 1], lon[RRR - 1]) < 0.0 and land_ocean(ETOPO, lat[RRR + 1],
                                                                                       lon[RRR + 1]) < 0.0):
                rsize[RRR] = np.nan
                lat[RRR] = np.nan
                lon[RRR] = np.nan
    for RRR in [0, len(time) - 1]:
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            rsize[RRR] = np.nan
            lat[RRR] = np.nan
            lon[RRR] = np.nan
            #
    if np.any(~np.isnan(rsize)):
        time_rsize_1st = np.where(~np.isnan(rsize))[0][0]
        time_rsize_lms = np.nanargmax(rsize)
        if (
                time_rsize_lms - time_rsize_1st) >= 4:  # and np.all(~np.isnan(rsize[time_rsize_1st:time_rsize_lms+1])): # at least obs of rsize of a day
            sn_out = sn[-1]
            year_out = year[-1]
            name_out = name[-1]
            lat_out = lat_withland
            lon_out = lon_withland
            lat_used = lat[time_rsize_1st:time_rsize_lms + 1]
            lon_used = lon[time_rsize_1st:time_rsize_lms + 1]
    return (sn_out, lat_out, lon_out, lat_used, lon_used, name_out, year_out)
    #


def find_size(data_in, SOURCE_LAT, SOURCE_LON, SOURCE_MAX_WIND, SOURCE_MIN_PRES, SOURCE_RSIZE_18_1, SOURCE_RSIZE_18_2,
              SOURCE_RSIZE_18_3, SOURCE_RSIZE_18_4, \
              SOURCE_RSIZE_26_1, SOURCE_RSIZE_26_2, SOURCE_RSIZE_26_3, SOURCE_RSIZE_26_4, \
              SOURCE_RSIZE_33_1, SOURCE_RSIZE_33_2, SOURCE_RSIZE_33_3, SOURCE_RSIZE_33_4):
    lat = np.asarray(data_in[SOURCE_LAT].values)
    lon = np.asarray(data_in[SOURCE_LON].values)
    vmax = np.asarray(data_in[SOURCE_MAX_WIND].values)
    pmin = np.asarray(data_in[SOURCE_MIN_PRES].values)
    # get size of interest (not for reference)
    tmp1 = np.asarray(data_in[SOURCE_RSIZE_18_1].values)
    tmp2 = np.asarray(data_in[SOURCE_RSIZE_18_2].values)
    tmp3 = np.asarray(data_in[SOURCE_RSIZE_18_3].values)
    tmp4 = np.asarray(data_in[SOURCE_RSIZE_18_4].values)
    tmp1[tmp1 <= 0] = np.nan
    tmp2[tmp2 <= 0] = np.nan
    tmp3[tmp3 <= 0] = np.nan
    tmp4[tmp4 <= 0] = np.nan
    rsize = np.nanmean([tmp1, tmp2, tmp3, tmp4], 0)
    # rsize      = (tmp1 + tmp2 + tmp3 + tmp4) / 4
    time = data_in['ISO_TIME'].values
    nature = data_in['NATURE'].values
    sn = data_in['SID'].values
    rsize_out = np.asarray([np.nan])
    rsize_out_3h = np.asarray([np.nan])
    CHANGErel = []
    CHANGErel_MARK = []
    CHANGErel_TIME = []
    CHANGEabs = []
    # remove inregular record not on 00 06 12 18
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][11:13]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lat = lat[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lon = lon[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    vmax = pmin[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    pmin = pmin[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    rsize = rsize[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    nature = nature[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    sn = sn[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    #
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][14:16]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00')]
    lat = lat[(tmp == '00')]
    lon = lon[(tmp == '00')]
    vmax = pmin[(tmp == '00')]
    pmin = pmin[(tmp == '00')]
    rsize = rsize[(tmp == '00')]
    nature = nature[(tmp == '00')]
    sn = sn[(tmp == '00')]
    # TC nature control, remove all extratropical period
    time = time[nature != 'ET']
    lat = lat[nature != 'ET']
    lon = lon[nature != 'ET']
    vmax = vmax[nature != 'ET']
    pmin = pmin[nature != 'ET']
    rsize = rsize[nature != 'ET']
    sn = sn[nature != 'ET']
    nature = nature[nature != 'ET']
    #
    # remove size over land
    for RRR in range(1, len(time) - 1):
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            if not (land_ocean(ETOPO, lat[RRR - 1], lon[RRR - 1]) < 0.0 and land_ocean(ETOPO, lat[RRR + 1],
                                                                                       lon[RRR + 1]) < 0.0):
                rsize[RRR] = np.nan
    for RRR in [0, len(time) - 1]:
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            rsize[RRR] = np.nan
    if np.any(~np.isnan(rsize)):
        time_rsize_1st = np.where(~np.isnan(rsize))[0][0]
        time_rsize_lms = np.nanargmax(rsize)
        if (
                time_rsize_lms - time_rsize_1st) >= 4:  # and np.all(~np.isnan(rsize[time_rsize_1st:time_rsize_lms+1])): # at least obs of rsize of a day
            print(sn[-1] + ' = ', end='')
            for TTT in range(time_rsize_1st + 1, time_rsize_lms + 1):
                print(time[TTT][11:13] + ' ', end='')
                time1 = utc2lst(time[TTT - 1], lon[TTT - 1])
                time2 = utc2lst(time[TTT], lon[TTT])
                HOUR_1_LST = int(time1[11:13])
                HOUR_2_LST = int(time2[11:13])
                HOUR_1 = time[TTT - 1][11:13]
                HOUR_2 = time[TTT][11:13]
                delta_rsize = (rsize[TTT] - rsize[TTT - 1]) / rsize[TTT - 1]
                CHANGErel.extend([delta_rsize])
                CHANGErel_TIME.extend([[HOUR_1_LST, HOUR_2_LST]])
                delta_rsize = (rsize[TTT] - rsize[TTT - 1])
                CHANGEabs.extend([delta_rsize])
                if HOUR_1_LST >= 6 and HOUR_1_LST <= 18 and HOUR_2_LST >= 6 and HOUR_2_LST <= 18:
                    CHANGErel_MARK.extend([0])
                elif (HOUR_1_LST >= 18 and (HOUR_2_LST >= 18 or HOUR_2_LST <= 6)) or (
                        HOUR_1_LST <= 6 and HOUR_2_LST <= 6):
                    CHANGErel_MARK.extend([1])
                # else:
                #   CHANGErel_MARK .extend([2])
                elif HOUR_1_LST < 6 and HOUR_2_LST > 6:
                    CHANGErel_MARK.extend([2])
                elif HOUR_1_LST < 18 and (HOUR_2_LST > 18 or (HOUR_2_LST >= 0 and HOUR_2_LST <= 2)):
                    CHANGErel_MARK.extend([3])
                else:
                    print('*************************************************************************************',
                          HOUR_1_LST, HOUR_2_LST)
            print('')
            rsize_out = rsize[time_rsize_1st:time_rsize_lms + 1]
            # interpolate 6-h data into 3-h interval
            rsize_out_3h = []
            rsize_out_3h.extend([rsize_out[0]])
            rsize_out_3h.extend([(rsize_out[0] + rsize_out[1]) / 2.])
            for rrr in range(1, len(rsize_out) - 1):
                rsize_out_3h.extend([rsize_out[rrr]])
                rsize_out_3h.extend([(rsize_out[rrr] + rsize_out[rrr + 1]) / 2.])
            rsize_out_3h.extend([rsize_out[-1]])
    return (CHANGErel, CHANGErel_MARK, rsize_out, rsize_out_3h, CHANGErel_TIME, CHANGEabs)
    #


def find_vmax(data_in, SOURCE_LAT, SOURCE_LON, SOURCE_MAX_WIND, SOURCE_MIN_PRES, SOURCE_RSIZE_18_1, SOURCE_RSIZE_18_2,
              SOURCE_RSIZE_18_3, SOURCE_RSIZE_18_4, \
              SOURCE_RSIZE_26_1, SOURCE_RSIZE_26_2, SOURCE_RSIZE_26_3, SOURCE_RSIZE_26_4, \
              SOURCE_RSIZE_33_1, SOURCE_RSIZE_33_2, SOURCE_RSIZE_33_3, SOURCE_RSIZE_33_4):
    lat = np.asarray(data_in[SOURCE_LAT].values)
    lon = np.asarray(data_in[SOURCE_LON].values)
    vmax = np.asarray(data_in[SOURCE_MAX_WIND].values)
    pmin = np.asarray(data_in[SOURCE_MIN_PRES].values)
    # get size of interest (not for reference)
    tmp1 = np.asarray(data_in[SOURCE_RSIZE_18_1].values)
    tmp2 = np.asarray(data_in[SOURCE_RSIZE_18_2].values)
    tmp3 = np.asarray(data_in[SOURCE_RSIZE_18_3].values)
    tmp4 = np.asarray(data_in[SOURCE_RSIZE_18_4].values)
    tmp1[tmp1 <= 0] = np.nan
    tmp2[tmp2 <= 0] = np.nan
    tmp3[tmp3 <= 0] = np.nan
    tmp4[tmp4 <= 0] = np.nan
    rsize = np.nanmean([tmp1, tmp2, tmp3, tmp4], 0)
    # rsize      = (tmp1 + tmp2 + tmp3 + tmp4) / 4
    time = data_in['ISO_TIME'].values
    nature = data_in['NATURE'].values
    sn = data_in['SID'].values
    vmax_out = np.asarray([np.nan])
    # remove inregular record not on 00 06 12 18
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][11:13]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lat = lat[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lon = lon[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    vmax = vmax[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    pmin = pmin[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    rsize = rsize[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    nature = nature[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    sn = sn[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    #
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][14:16]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00')]
    lat = lat[(tmp == '00')]
    lon = lon[(tmp == '00')]
    vmax = vmax[(tmp == '00')]
    pmin = pmin[(tmp == '00')]
    rsize = rsize[(tmp == '00')]
    nature = nature[(tmp == '00')]
    sn = sn[(tmp == '00')]
    # TC nature control, remove all extratropical period
    time = time[nature != 'ET']
    lat = lat[nature != 'ET']
    lon = lon[nature != 'ET']
    vmax = vmax[nature != 'ET']
    pmin = pmin[nature != 'ET']
    rsize = rsize[nature != 'ET']
    sn = sn[nature != 'ET']
    nature = nature[nature != 'ET']
    #
    # remove size over land
    for RRR in range(1, len(time) - 1):
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            if not (land_ocean(ETOPO, lat[RRR - 1], lon[RRR - 1]) < 0.0 and land_ocean(ETOPO, lat[RRR + 1],
                                                                                       lon[RRR + 1]) < 0.0):
                rsize[RRR] = np.nan
                lat[RRR] = np.nan
                lon[RRR] = np.nan
    for RRR in [0, len(time) - 1]:
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            rsize[RRR] = np.nan
            lat[RRR] = np.nan
            lon[RRR] = np.nan
            #
    if np.any(~np.isnan(rsize)):
        time_rsize_1st = np.where(~np.isnan(rsize))[0][0]
        time_rsize_lms = np.nanargmax(rsize)
        if (
                time_rsize_lms - time_rsize_1st) >= 4:  # and np.all(~np.isnan(rsize[time_rsize_1st:time_rsize_lms+1])): # at least obs of rsize of a day
            vmax_out = vmax[time_rsize_1st:time_rsize_lms + 1]
    return (vmax_out, vmax_out)


def find_rain(data_in, SOURCE_LAT, SOURCE_LON, SOURCE_MAX_WIND, SOURCE_MIN_PRES, SOURCE_RSIZE_18_1, SOURCE_RSIZE_18_2,
              SOURCE_RSIZE_18_3, SOURCE_RSIZE_18_4, \
              SOURCE_RSIZE_26_1, SOURCE_RSIZE_26_2, SOURCE_RSIZE_26_3, SOURCE_RSIZE_26_4, \
              SOURCE_RSIZE_33_1, SOURCE_RSIZE_33_2, SOURCE_RSIZE_33_3, SOURCE_RSIZE_33_4):
    lat = np.asarray(data_in[SOURCE_LAT].values)
    lon = np.asarray(data_in[SOURCE_LON].values)
    vmax = np.asarray(data_in[SOURCE_MAX_WIND].values)
    pmin = np.asarray(data_in[SOURCE_MIN_PRES].values)
    # get size of interest (not for reference)
    tmp1 = np.asarray(data_in[SOURCE_RSIZE_18_1].values)
    tmp2 = np.asarray(data_in[SOURCE_RSIZE_18_2].values)
    tmp3 = np.asarray(data_in[SOURCE_RSIZE_18_3].values)
    tmp4 = np.asarray(data_in[SOURCE_RSIZE_18_4].values)
    tmp1[tmp1 <= 0] = np.nan
    tmp2[tmp2 <= 0] = np.nan
    tmp3[tmp3 <= 0] = np.nan
    tmp4[tmp4 <= 0] = np.nan
    rsize = np.nanmean([tmp1, tmp2, tmp3, tmp4], 0)
    # rsize      = (tmp1 + tmp2 + tmp3 + tmp4) / 4
    time = data_in['ISO_TIME'].values
    nature = data_in['NATURE'].values
    sn = data_in['SID'].values
    # this PROFILE_GRIDs should be matched up with the number in function find_rain_profile
    PROFILE_GRIDs = 40
    rain_profile_out_mean = np.nan * np.zeros(PROFILE_GRIDs + 1)
    rain_profile_out_3h = np.nan * np.zeros(PROFILE_GRIDs + 1)
    # remove inregular record not on 00 06 12 18
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][11:13]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lat = lat[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lon = lon[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    vmax = vmax[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    pmin = pmin[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    rsize = rsize[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    nature = nature[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    sn = sn[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    #
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][14:16]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00')]
    lat = lat[(tmp == '00')]
    lon = lon[(tmp == '00')]
    vmax = pmin[(tmp == '00')]
    pmin = pmin[(tmp == '00')]
    rsize = rsize[(tmp == '00')]
    nature = nature[(tmp == '00')]
    sn = sn[(tmp == '00')]
    # TC nature control, remove all extratropical period
    time = time[nature != 'ET']
    lat = lat[nature != 'ET']
    lon = lon[nature != 'ET']
    vmax = vmax[nature != 'ET']
    pmin = pmin[nature != 'ET']
    rsize = rsize[nature != 'ET']
    sn = sn[nature != 'ET']
    nature = nature[nature != 'ET']
    #
    # remove size over land
    for RRR in range(1, len(time) - 1):
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            if not (land_ocean(ETOPO, lat[RRR - 1], lon[RRR - 1]) < 0.0 and land_ocean(ETOPO, lat[RRR + 1],
                                                                                       lon[RRR + 1]) < 0.0):
                rsize[RRR] = np.nan
                lat[RRR] = np.nan
                lon[RRR] = np.nan
    for RRR in [0, len(time) - 1]:
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            rsize[RRR] = np.nan
            lat[RRR] = np.nan
            lon[RRR] = np.nan
            #
    if np.any(~np.isnan(rsize)):
        time_rsize_1st = np.where(~np.isnan(rsize))[0][0]
        time_rsize_lms = np.nanargmax(rsize)
        if (
                time_rsize_lms - time_rsize_1st) >= 4:  # and np.all(~np.isnan(rsize[time_rsize_1st:time_rsize_lms+1])): # at least obs of rsize of a day
            for TTT in range(time_rsize_1st + 1, time_rsize_lms + 1):
                HOUR_1 = time[TTT - 1][11:13]
                HOUR_2 = time[TTT][11:13]
                time0 = datetime.strptime(time[TTT - 1], "%Y-%m-%d %H:%M:%S")
                # 3 hour in between the two BT records
                if HOUR_1 == '00':
                    HOUR_between = '03'
                if HOUR_1 == '06':
                    HOUR_between = '09'
                if HOUR_1 == '12':
                    HOUR_between = '15'
                if HOUR_1 == '18':
                    HOUR_between = '21'
                lon_between = (lon[TTT - 1] + lon[TTT - 0]) / 2.
                lat_between = (lat[TTT - 1] + lat[TTT - 0]) / 2.
                rsize_between = (rsize[TTT - 1] + rsize[TTT - 0]) / 2.
                time_between = datetime.strftime(time0 + timedelta(0, 3600 * 3), "%Y-%m-%d %H:%M:%S")
                if TTT == time_rsize_1st + 1:
                    rain_profile_out_mean = []
                    rain_profile_out_3h = []
                    rainprofile_1 = find_trmm_profile(time[TTT - 1], lat[TTT - 1], lon[TTT - 1], rsize[TTT - 1])
                    rain_profile_out_3h.append(rainprofile_1)
                rainprofile_2 = find_trmm_profile(time[TTT - 0], lat[TTT - 0], lon[TTT - 0], rsize[TTT - 0])
                rainprofile_between = find_trmm_profile(time_between, lat_between, lon_between, rsize_between)
                rain_profile_out_3h.append(rainprofile_between)
                rain_profile_out_3h.append(rainprofile_2)
                rain_profile_out_mean.append(np.mean(rain_profile_out_3h[-3:], 0))
    return (rain_profile_out_mean, rain_profile_out_3h)


def find_btemp(data_in, SOURCE_LAT, SOURCE_LON, SOURCE_MAX_WIND, SOURCE_MIN_PRES, SOURCE_RSIZE_18_1, SOURCE_RSIZE_18_2,
               SOURCE_RSIZE_18_3, SOURCE_RSIZE_18_4, \
               SOURCE_RSIZE_26_1, SOURCE_RSIZE_26_2, SOURCE_RSIZE_26_3, SOURCE_RSIZE_26_4, \
               SOURCE_RSIZE_33_1, SOURCE_RSIZE_33_2, SOURCE_RSIZE_33_3, SOURCE_RSIZE_33_4):
    lat = np.asarray(data_in[SOURCE_LAT].values)
    lon = np.asarray(data_in[SOURCE_LON].values)
    vmax = np.asarray(data_in[SOURCE_MAX_WIND].values)
    pmin = np.asarray(data_in[SOURCE_MIN_PRES].values)
    # get size of interest (not for reference)
    tmp1 = np.asarray(data_in[SOURCE_RSIZE_18_1].values)
    tmp2 = np.asarray(data_in[SOURCE_RSIZE_18_2].values)
    tmp3 = np.asarray(data_in[SOURCE_RSIZE_18_3].values)
    tmp4 = np.asarray(data_in[SOURCE_RSIZE_18_4].values)
    tmp1[tmp1 <= 0] = np.nan
    tmp2[tmp2 <= 0] = np.nan
    tmp3[tmp3 <= 0] = np.nan
    tmp4[tmp4 <= 0] = np.nan
    rsize = np.nanmean([tmp1, tmp2, tmp3, tmp4], 0)
    # rsize      = (tmp1 + tmp2 + tmp3 + tmp4) / 4
    time = data_in['ISO_TIME'].values
    nature = data_in['NATURE'].values
    sn = data_in['SID'].values
    btemp_out = np.asarray([np.nan])
    hour = np.asarray([np.nan])
    hour_3h = np.asarray([np.nan])
    # this PROFILE_GRIDs should be matched up with the number in function find_hursat_btemp
    PROFILE_GRIDs = 151
    btemp_profile_out_mean = np.nan * np.zeros(PROFILE_GRIDs)
    btemp_profile_out_3h = np.nan * np.zeros(PROFILE_GRIDs)
    # remove inregular record not on 00 06 12 18
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][11:13]])
    tmp = np.asarray(tmp)
    hour = copy.copy(tmp)
    hour = hour[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    time = time[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lat = lat[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    lon = lon[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    vmax = vmax[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    pmin = pmin[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    rsize = rsize[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    nature = nature[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    sn = sn[(tmp == '00') | (tmp == '06') | (tmp == '12') | (tmp == '18')]
    #
    tmp = []
    for RRR in range(len(time)):
        tmp.extend([time[RRR][14:16]])
    tmp = np.asarray(tmp)
    time = time[(tmp == '00')]
    lat = lat[(tmp == '00')]
    lon = lon[(tmp == '00')]
    vmax = pmin[(tmp == '00')]
    pmin = pmin[(tmp == '00')]
    rsize = rsize[(tmp == '00')]
    nature = nature[(tmp == '00')]
    sn = sn[(tmp == '00')]
    # TC nature control, remove all extratropical period
    time = time[nature != 'ET']
    lat = lat[nature != 'ET']
    lon = lon[nature != 'ET']
    vmax = vmax[nature != 'ET']
    pmin = pmin[nature != 'ET']
    rsize = rsize[nature != 'ET']
    sn = sn[nature != 'ET']
    nature = nature[nature != 'ET']
    #
    # remove size over land
    for RRR in range(1, len(time) - 1):
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            if not (land_ocean(ETOPO, lat[RRR - 1], lon[RRR - 1]) < 0.0 and land_ocean(ETOPO, lat[RRR + 1],
                                                                                       lon[RRR + 1]) < 0.0):
                rsize[RRR] = np.nan
                lat[RRR] = np.nan
                lon[RRR] = np.nan
    for RRR in [0, len(time) - 1]:
        if land_ocean(ETOPO, lat[RRR], lon[RRR]) >= 0.0:
            rsize[RRR] = np.nan
            lat[RRR] = np.nan
            lon[RRR] = np.nan
            #
    if np.any(~np.isnan(rsize)):
        time_rsize_1st = np.where(~np.isnan(rsize))[0][0]
        time_rsize_lms = np.nanargmax(rsize)
        if (
                time_rsize_lms - time_rsize_1st) >= 4:  # and np.all(~np.isnan(rsize[time_rsize_1st:time_rsize_lms+1])): # at least obs of rsize of a day
            for TTT in range(time_rsize_1st + 1, time_rsize_lms + 1):
                HOUR_1 = time[TTT - 1][11:13]
                HOUR_2 = time[TTT][11:13]
                time0 = datetime.strptime(time[TTT - 1], "%Y-%m-%d %H:%M:%S")
                # 3 hour in between the two BT records
                if HOUR_1 == '00':
                    HOUR_between = '03'
                if HOUR_1 == '06':
                    HOUR_between = '09'
                if HOUR_1 == '12':
                    HOUR_between = '15'
                if HOUR_1 == '18':
                    HOUR_between = '21'
                lon_between = (lon[TTT - 1] + lon[TTT - 0]) / 2.
                lat_between = (lat[TTT - 1] + lat[TTT - 0]) / 2.
                rsize_between = (rsize[TTT - 1] + rsize[TTT - 0]) / 2.
                time_between = datetime.strftime(time0 + timedelta(0, 3600 * 3), "%Y-%m-%d %H:%M:%S")
                # find the brightness tempature from HURSAT
                btemp_1_tmp = find_hursat_btemp(time[TTT - 1], lat[TTT - 1], lon[TTT - 1], rsize[TTT - 1], sn[TTT - 1])
                btemp_2_tmp = find_hursat_btemp(time[TTT - 0], lat[TTT - 0], lon[TTT - 0], rsize[TTT - 0], sn[TTT - 0])
                btemp_1 = btemp_1_tmp[0]
                btemp_2 = btemp_2_tmp[0]
                btemp_1_profile = btemp_1_tmp[1]
                btemp_2_profile = btemp_2_tmp[1]
                if TTT == time_rsize_1st + 1:
                    btemp_profile_out_3h = []
                    btemp_profile_out_mean = []
                    btemp_profile_out_3h.append(btemp_1_profile)
                btemp_between_tmp = find_hursat_btemp(time_between, lat_between, lon_between, rsize_between,
                                                      sn[TTT - 0])
                btemp_between_profile = btemp_between_tmp[1]
                btemp_profile_out_3h.append(btemp_between_profile)
                btemp_profile_out_3h.append(btemp_2_profile)
                btemp_profile_out_mean.append(np.mean(btemp_profile_out_3h[-3:], 0))
    return (btemp_profile_out_mean, btemp_profile_out_3h)


# define a new operator "x=>y" taking y=nan as y=0
def larger_with_nan(x, y):
    output = False
    if x >= y or np.isnan(y):
        output = True
    return output


# filter the years first to save computaional time in the next large loop
ibt_within_years = ibt.groupby(['SID']).filter(
    lambda x: x['SEASON'].values[0] >= START_YEAR and x['SEASON'].values[0] <= END_YEAR)
# loop through years
# CASE[YEAR][BASIN][CASE][VAR]
# CASE_GB[YEAR][CASE][VAR]
SN_VAR = []
SIZE_VAR = []
VMAX_VAR = []
RAIN_VAR = []
BTEMP_VAR = []
for YEAR in INDEX_year:
    SN_VAR.append([])
    SIZE_VAR.append([])
    VMAX_VAR.append([])
    RAIN_VAR.append([])
    BTEMP_VAR.append([])
    # filter for the year first
    ibt_in_one_year = ibt_within_years.groupby(['SID']).filter(lambda x: x['SEASON'].values[0] == YEAR)
    # loop through basins
    for BASIN, BASIN_NAME in enumerate(BASIN_NAMEs[0:-3]):  # enumerate(BASIN_NAMEs[0:-3]):
        print(YEAR, BASIN_NAME)
        SN_VAR[-1].append([])
        SIZE_VAR[-1].append([])
        VMAX_VAR[-1].append([])
        RAIN_VAR[-1].append([])
        BTEMP_VAR[-1].append([])
        # one cyclone may have records in two angency reports, eg in the middle of pacific
        # tmp_other_basin helps to find the agency reporting the lifetime max intensity.
        # then define the cyclone is in the corresponding basin
        tmp_other_basin = [0, 1, 2, 3, 4, 5, 6]
        # remove the current looping basin
        tmp_other_basin.pop(BASIN)
        filtered_ibtracs = ibt_in_one_year.groupby(['SID']).filter(lambda x: \
                                                                       x[AGENCY_MAX_WIND[BASIN]].max() >= CAT_KT.loc[
                                                                           ('LOWER'), ('TS')] \
                                                                       and x['BASIN'].values[np.nanargmax(
                                                                           x['USA_WIND'].values)] == BASIN_NAME)
        #        tmp = filtered_ibtracs.groupby(['SID'])['SID','ISO_TIME','NATURE','NAME','SEASON', \
        #              AGENCY_LAT[BASIN],AGENCY_LON[BASIN],AGENCY_MAX_WIND[BASIN],AGENCY_MIN_PRES[BASIN],AGENCY_R18_NE[BASIN],AGENCY_R18_SE[BASIN],AGENCY_R18_SW[BASIN],AGENCY_R18_NW[BASIN],\
        #                                                                                                AGENCY_R26_NE[BASIN],AGENCY_R26_SE[BASIN],AGENCY_R26_SW[BASIN],AGENCY_R26_NW[BASIN],\
        #                                                                                                AGENCY_R33_NE[BASIN],AGENCY_R33_SE[BASIN],AGENCY_R33_SW[BASIN],AGENCY_R33_NW[BASIN]] \
        #              .apply(find_sn         ,SOURCE_LAT      = AGENCY_LAT     [BASIN],\
        #                                      SOURCE_LON      = AGENCY_LON     [BASIN],\
        #                                      SOURCE_MAX_WIND = AGENCY_MAX_WIND[BASIN],\
        #                                      SOURCE_MIN_PRES = AGENCY_MIN_PRES[BASIN],\
        #                                      SOURCE_RSIZE_18_1    = AGENCY_R18_NE  [BASIN],\
        #                                      SOURCE_RSIZE_18_2    = AGENCY_R18_SE  [BASIN],\
        #                                      SOURCE_RSIZE_18_3    = AGENCY_R18_SW  [BASIN],\
        #                                      SOURCE_RSIZE_18_4    = AGENCY_R18_NW  [BASIN],\
        #                                      SOURCE_RSIZE_26_1    = AGENCY_R26_NE  [BASIN],\
        #                                      SOURCE_RSIZE_26_2    = AGENCY_R26_SE  [BASIN],\
        #                                      SOURCE_RSIZE_26_3    = AGENCY_R26_SW  [BASIN],\
        #                                      SOURCE_RSIZE_26_4    = AGENCY_R26_NW  [BASIN],\
        #                                      SOURCE_RSIZE_33_1    = AGENCY_R33_NE  [BASIN],\
        #                                      SOURCE_RSIZE_33_2    = AGENCY_R33_SE  [BASIN],\
        #                                      SOURCE_RSIZE_33_3    = AGENCY_R33_SW  [BASIN],\
        #                                      SOURCE_RSIZE_33_4    = AGENCY_R33_NW  [BASIN])
        #        SN_VAR[-1][-1].extend(tmp.values)
        #        tmp = filtered_ibtracs.groupby(['SID'])['SID','ISO_TIME','NATURE', \
        #              AGENCY_LAT[BASIN],AGENCY_LON[BASIN],AGENCY_MAX_WIND[BASIN],AGENCY_MIN_PRES[BASIN],AGENCY_R18_NE[BASIN],AGENCY_R18_SE[BASIN],AGENCY_R18_SW[BASIN],AGENCY_R18_NW[BASIN],\
        #                                                                                                AGENCY_R26_NE[BASIN],AGENCY_R26_SE[BASIN],AGENCY_R26_SW[BASIN],AGENCY_R26_NW[BASIN],\
        #                                                                                                AGENCY_R33_NE[BASIN],AGENCY_R33_SE[BASIN],AGENCY_R33_SW[BASIN],AGENCY_R33_NW[BASIN]] \
        #              .apply(find_size       ,SOURCE_LAT      = AGENCY_LAT     [BASIN],\
        #                                      SOURCE_LON      = AGENCY_LON     [BASIN],\
        #                                      SOURCE_MAX_WIND = AGENCY_MAX_WIND[BASIN],\
        #                                      SOURCE_MIN_PRES = AGENCY_MIN_PRES[BASIN],\
        #                                      SOURCE_RSIZE_18_1    = AGENCY_R18_NE  [BASIN],\
        #                                      SOURCE_RSIZE_18_2    = AGENCY_R18_SE  [BASIN],\
        #                                      SOURCE_RSIZE_18_3    = AGENCY_R18_SW  [BASIN],\
        #                                      SOURCE_RSIZE_18_4    = AGENCY_R18_NW  [BASIN],\
        #                                      SOURCE_RSIZE_26_1    = AGENCY_R26_NE  [BASIN],\
        #                                      SOURCE_RSIZE_26_2    = AGENCY_R26_SE  [BASIN],\
        #                                      SOURCE_RSIZE_26_3    = AGENCY_R26_SW  [BASIN],\
        #                                      SOURCE_RSIZE_26_4    = AGENCY_R26_NW  [BASIN],\
        #                                      SOURCE_RSIZE_33_1    = AGENCY_R33_NE  [BASIN],\
        #                                      SOURCE_RSIZE_33_2    = AGENCY_R33_SE  [BASIN],\
        #                                      SOURCE_RSIZE_33_3    = AGENCY_R33_SW  [BASIN],\
        #                                      SOURCE_RSIZE_33_4    = AGENCY_R33_NW  [BASIN]   )
        #        SIZE_VAR[-1][-1].extend(tmp.values)
        #        tmp = filtered_ibtracs.groupby(['SID'])['SID','ISO_TIME','NATURE', \
        #              AGENCY_LAT[BASIN],AGENCY_LON[BASIN],AGENCY_MAX_WIND[BASIN],AGENCY_MIN_PRES[BASIN],AGENCY_R18_NE[BASIN],AGENCY_R18_SE[BASIN],AGENCY_R18_SW[BASIN],AGENCY_R18_NW[BASIN],\
        #                                                                                                AGENCY_R26_NE[BASIN],AGENCY_R26_SE[BASIN],AGENCY_R26_SW[BASIN],AGENCY_R26_NW[BASIN],\
        #                                                                                                AGENCY_R33_NE[BASIN],AGENCY_R33_SE[BASIN],AGENCY_R33_SW[BASIN],AGENCY_R33_NW[BASIN]] \
        #              .apply(find_vmax       ,SOURCE_LAT      = AGENCY_LAT     [BASIN],\
        #                                      SOURCE_LON      = AGENCY_LON     [BASIN],\
        #                                      SOURCE_MAX_WIND = AGENCY_MAX_WIND[BASIN],\
        #                                      SOURCE_MIN_PRES = AGENCY_MIN_PRES[BASIN],\
        #                                      SOURCE_RSIZE_18_1    = AGENCY_R18_NE  [BASIN],\
        #                                      SOURCE_RSIZE_18_2    = AGENCY_R18_SE  [BASIN],\
        #                                      SOURCE_RSIZE_18_3    = AGENCY_R18_SW  [BASIN],\
        #                                      SOURCE_RSIZE_18_4    = AGENCY_R18_NW  [BASIN],\
        #                                      SOURCE_RSIZE_26_1    = AGENCY_R26_NE  [BASIN],\
        #                                      SOURCE_RSIZE_26_2    = AGENCY_R26_SE  [BASIN],\
        #                                      SOURCE_RSIZE_26_3    = AGENCY_R26_SW  [BASIN],\
        #                                      SOURCE_RSIZE_26_4    = AGENCY_R26_NW  [BASIN],\
        #                                      SOURCE_RSIZE_33_1    = AGENCY_R33_NE  [BASIN],\
        #                                      SOURCE_RSIZE_33_2    = AGENCY_R33_SE  [BASIN],\
        #                                      SOURCE_RSIZE_33_3    = AGENCY_R33_SW  [BASIN],\
        #                                      SOURCE_RSIZE_33_4    = AGENCY_R33_NW  [BASIN]   )
        #        VMAX_VAR[-1][-1].extend(tmp.values)
        #        tmp = filtered_ibtracs.groupby(['SID'])['SID','ISO_TIME','NATURE', \
        #              AGENCY_LAT[BASIN],AGENCY_LON[BASIN],AGENCY_MAX_WIND[BASIN],AGENCY_MIN_PRES[BASIN],AGENCY_R18_NE[BASIN],AGENCY_R18_SE[BASIN],AGENCY_R18_SW[BASIN],AGENCY_R18_NW[BASIN],\
        #                                                                                                AGENCY_R26_NE[BASIN],AGENCY_R26_SE[BASIN],AGENCY_R26_SW[BASIN],AGENCY_R26_NW[BASIN],\
        #                                                                                                AGENCY_R33_NE[BASIN],AGENCY_R33_SE[BASIN],AGENCY_R33_SW[BASIN],AGENCY_R33_NW[BASIN]] \
        #              .apply(find_rain       ,SOURCE_LAT      = AGENCY_LAT     [BASIN],\
        #                                      SOURCE_LON      = AGENCY_LON     [BASIN],\
        #                                      SOURCE_MAX_WIND = AGENCY_MAX_WIND[BASIN],\
        #                                      SOURCE_MIN_PRES = AGENCY_MIN_PRES[BASIN],\
        #                                      SOURCE_RSIZE_18_1    = AGENCY_R18_NE  [BASIN],\
        #                                      SOURCE_RSIZE_18_2    = AGENCY_R18_SE  [BASIN],\
        #                                      SOURCE_RSIZE_18_3    = AGENCY_R18_SW  [BASIN],\
        #                                      SOURCE_RSIZE_18_4    = AGENCY_R18_NW  [BASIN],\
        #                                      SOURCE_RSIZE_26_1    = AGENCY_R26_NE  [BASIN],\
        #                                      SOURCE_RSIZE_26_2    = AGENCY_R26_SE  [BASIN],\
        #                                      SOURCE_RSIZE_26_3    = AGENCY_R26_SW  [BASIN],\
        #                                      SOURCE_RSIZE_26_4    = AGENCY_R26_NW  [BASIN],\
        #                                      SOURCE_RSIZE_33_1    = AGENCY_R33_NE  [BASIN],\
        #                                      SOURCE_RSIZE_33_2    = AGENCY_R33_SE  [BASIN],\
        #                                      SOURCE_RSIZE_33_3    = AGENCY_R33_SW  [BASIN],\
        #                                      SOURCE_RSIZE_33_4    = AGENCY_R33_NW  [BASIN]   )
        #        RAIN_VAR[-1][-1].extend(tmp.values)
        tmp = filtered_ibtracs.groupby(['SID'])['SID', 'ISO_TIME', 'NATURE', \
                                                AGENCY_LAT[BASIN], AGENCY_LON[BASIN], AGENCY_MAX_WIND[BASIN],
                                                AGENCY_MIN_PRES[BASIN], AGENCY_R18_NE[BASIN], AGENCY_R18_SE[BASIN],
                                                AGENCY_R18_SW[BASIN], AGENCY_R18_NW[BASIN], \
                                                AGENCY_R26_NE[BASIN], AGENCY_R26_SE[BASIN], AGENCY_R26_SW[BASIN],
                                                AGENCY_R26_NW[BASIN], \
                                                AGENCY_R33_NE[BASIN], AGENCY_R33_SE[BASIN], AGENCY_R33_SW[BASIN],
                                                AGENCY_R33_NW[BASIN]] \
            .apply(find_btemp, SOURCE_LAT=AGENCY_LAT[BASIN], \
                   SOURCE_LON=AGENCY_LON[BASIN], \
                   SOURCE_MAX_WIND=AGENCY_MAX_WIND[BASIN], \
                   SOURCE_MIN_PRES=AGENCY_MIN_PRES[BASIN], \
                   SOURCE_RSIZE_18_1=AGENCY_R18_NE[BASIN], \
                   SOURCE_RSIZE_18_2=AGENCY_R18_SE[BASIN], \
                   SOURCE_RSIZE_18_3=AGENCY_R18_SW[BASIN], \
                   SOURCE_RSIZE_18_4=AGENCY_R18_NW[BASIN], \
                   SOURCE_RSIZE_26_1=AGENCY_R26_NE[BASIN], \
                   SOURCE_RSIZE_26_2=AGENCY_R26_SE[BASIN], \
                   SOURCE_RSIZE_26_3=AGENCY_R26_SW[BASIN], \
                   SOURCE_RSIZE_26_4=AGENCY_R26_NW[BASIN], \
                   SOURCE_RSIZE_33_1=AGENCY_R33_NE[BASIN], \
                   SOURCE_RSIZE_33_2=AGENCY_R33_SE[BASIN], \
                   SOURCE_RSIZE_33_3=AGENCY_R33_SW[BASIN], \
                   SOURCE_RSIZE_33_4=AGENCY_R33_NW[BASIN])
        BTEMP_VAR[-1][-1].extend(tmp.values)
#        #
DataOutPath = '/home/sw1013/GROWTH_RATE_DERIVE/data/'
# np.save(DataOutPath +'ALL_R18_SN_VAR_withzero'        ,SN_VAR)
# np.save(DataOutPath +'ALL_R18_SIZE_VAR_withzero'      ,SIZE_VAR)
# np.save(DataOutPath +'ALL_R18_VMAX_VAR_withzero'      ,VMAX_VAR)
# np.save(DataOutPath +'ALL_R18_RAIN_VAR_pcp_withzero'  ,RAIN_VAR)
# np.save(DataOutPath +'ALL_R18_BTEMP_VAR_withzero'     ,BTEMP_VAR)


# ----- this is a land filter to remove land within a distance (degree)
###def land_ocean(ETOPO,lat_in,lon_in,dis_consider):
###   #Creat ETOPO that contains the land depth in a unit of meter
###   #To call, use ETOPO.loc[LAT,LON], LAT and LON should be float
###   #< 0 is water, and >=0 is land
###   # unit of dis_consider is degree
###   # land_ocean TRUE = all in ocean, FALSE = at least one above sea level
###   lat_ind      = np.argmin(np.abs(GRIDS_LAT-lat_in))
###   lon_ind      = np.argmin(np.abs(GRIDS_LON-lon_in))
###   DIS          = np.sqrt((INDEX_LON-lon_in)**2+(INDEX_LAT-lat_in)**2)
###   tmpETOPO     = copy.copy(ETOPO.values)
###   tmpETOPO[DIS>=dis_consider] = np.nan
###   if np.sum(tmpETOPO>=0)==0: # all ocean, none point above mean sea level
###      land_ocean   = True
###   else:
###      land_ocean   = False
###   return land_ocean
