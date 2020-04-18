import atexit
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

WEBSERVER_QUERY_URL = "http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices"
FILE_LOAD_LIMIT = os.environ.get("FILE_LOAD_LIMIT", 8)
SEARCH_FOR_FILES = "/searchForFiles"
GET_FILE_URLS = "/getFileUrls"
MODIS_DOWNLOAD_URL = "http://www.sp.ph.ic.ac.uk/~erg10/safe/subset"

if "CACHE_DIRECTORY" in os.environ:
    LAADS_CACHE_PATH = os.path.join(os.environ["CACHE_DIRECTORY"], "laads_cache.csv")
    if os.path.isfile(LAADS_CACHE_PATH):
        LAADS_CACHE = pd.read_csv(LAADS_CACHE_PATH)
    else:
        LAADS_CACHE = pd.DataFrame(columns=["request", "response", "files"])


    @atexit.register
    def save_cache():
        LAADS_CACHE.to_csv(LAADS_CACHE_PATH,index=False)


def download_files_from_server(root_dir, file_urls, ignore_errors=False, include_headers=True):
    fpath = [os.path.join(root_dir, os.path.split(f)[1]) for f in file_urls]

    for i, file in enumerate(fpath):
        if os.path.isfile(file):
            print(f"{file} already downloaded... skipping...")
            continue
        print(f"Downloading {file_urls[i]}")
        # Begin downloading from the content server
        # Use wget for large files
        # headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        if include_headers:
            headers = {"Authorization": f"Bearer {os.environ['LAADS_API_KEY']}"}
        else:
            headers = None
        download = requests.get(file_urls[i], headers=headers,
                                stream=True)
        if download.status_code == 200:
            import sys
            with open(file, "wb") as f:
                length = download.headers.get("content-length")
                if length is None:
                    f.write(download.content)
                else:
                    d_l = 0
                    t_l = int(length)
                    for data in download.iter_content(chunk_size=4096):
                        d_l += len(data)
                        f.write(data)
                        done = int(50 * d_l / t_l)
                        sys.stdout.write("\r[%s%s] (%s %%)" % ('=' * done, ' ' * (50 - done), int(100 * d_l / t_l)))
                        sys.stdout.flush()
            print("\n")
        else:
            if ignore_errors:
                continue
            else:
                raise ConnectionError(f"Download failed with code:{download.status_code}")
        if not os.path.isfile(file):
            raise ConnectionError(
                f"File {file_urls[i]} could not be downloaded to {file}. Please check your internet connection and try again!")
    return fpath


def get_all_modis_data(root_dir, year_range=(2012, 2017)):
    for i in range(*year_range):
        dir = os.path.join(root_dir, str(i))
        Path(dir).mkdir(parents=True)
        files = [f"{MODIS_DOWNLOAD_URL}/{i}/new.{str(j).zfill(3)}.c61.nc" for j in range(1, 366)]
        download_files_from_server(dir, files, include_headers=False, ignore_errors=True)


def get_aerosol_data(root_dir, start_time, end_time, north, south, east, west):
    assert "LAADS_API_KEY" in os.environ
    query_response = requests.get(WEBSERVER_QUERY_URL + SEARCH_FOR_FILES, params={
        "collection": "5110",
        "products": "AERDT_L2_VIIRS_SNPP",
        "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "north": north,
        "south": south,
        "east": east,
        "west": west,
        "coordsOrTiles": "coords",
        "dayNightBoth": "D"
    })
    fileIdStr = ""
    root_fids = ET.fromstring(query_response.content)
    for child in root_fids:
        if child.text == "No results":
            raise FileNotFoundError
        fileIdStr += child.text + ","
    file_name_response = requests.get(WEBSERVER_QUERY_URL + GET_FILE_URLS, params={"fileIds": fileIdStr})
    if file_name_response.status_code != 200:
        raise ConnectionError(file_name_response.content)
    files = [f.text for f in ET.fromstring(file_name_response.content)]
    return download_files_from_server(root_dir, files)


def get_data(root_dir, start_time, end_time, include_mod=False, north=90, south=-90, west=-180, east=180,
             collection="5110",
             dayOrNight="DNB", ):
    '''
    Use SatPy to check if data exists already in root dir. If not contact the NASA LAADS DAC server to download the required data.
    '''

    # TODO: Is there away to get the filepath without two requests server? Doesn't seem to be be studying the API documentation

    products = "VNP02IMG,VNP03IMG"
    if include_mod:
        products += ",VNP02MOD,VNP03MOD"

    assert 'LAADS_API_KEY' in os.environ
    params = {
        "collection": collection,
        "products": products,
        "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "north": north,
        "south": south,
        "east": east,
        "west": west,
        "coordsOrTiles": "coords",
        "dayNightBoth": dayOrNight
    }
    try:
        query_response = requests.get(WEBSERVER_QUERY_URL + SEARCH_FOR_FILES, params=params)

        fileIdStr = ""
        root_fids = ET.fromstring(query_response.content)
        for child in root_fids:
            if child.text == "No results":
                raise FileNotFoundError
            fileIdStr += child.text + ","
        file_name_response = requests.get(WEBSERVER_QUERY_URL + GET_FILE_URLS, params={"fileIds": fileIdStr})
        if file_name_response.status_code != 200:
            raise ConnectionError(file_name_response.content)
        files = [f.text for f in ET.fromstring(file_name_response.content)]
        global LAADS_CACHE
        if not (LAADS_CACHE["request"] == params).any():
            LAADS_CACHE = LAADS_CACHE.append({"request": params, "response": files,
                                              "local": [os.path.join(root_dir, os.path.split(f)[1]) for f in files]},
                                             ignore_index=True)
        if len(files) > FILE_LOAD_LIMIT:
            raise RuntimeError("Too many files loaded to process efficiently")
    except ConnectionError as e:
        if (LAADS_CACHE["request"] == params).any():
            row = LAADS_CACHE.loc[LAADS_CACHE["request"] == params]
            return row["response"], row["files"]
        raise e
    return download_files_from_server(root_dir, files), files


