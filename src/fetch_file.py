import os
import xml.etree.ElementTree as ET

import requests

WEBSERVER_QUERY_URL = "http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices"

SEARCH_FOR_FILES = "/searchForFiles"
GET_FILE_URLS = "/getFileUrls"


def download_files_from_server(root_dir, file_urls):
    fpath = [os.path.join(root_dir, os.path.split(f)[1]) for f in file_urls]
    for i, file in enumerate(fpath):
        if os.path.isfile(file):
            print(f"{file} already downloaded... skipping...")
            continue
        print(f"Downloading {file_urls[i]}")
        # Begin downloading from the content server
        # Use wget for large files
        # headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        download = requests.get(file_urls[i], headers={"Authorization": f"Bearer {os.environ['LAADS_API_KEY']}"})
        if download.status_code == 200:
            open(file, "wb").write(download.content)
        else:
            raise ConnectionError()
        if not os.path.isfile(file):
            raise ConnectionError(
                f"File {file_urls[i]} could not be downloaded to {file}. Please check your internet connection and try again!")
    return fpath


def get_data(root_dir, start_time, end_time, north=90, south=-90, west=-180, east=180, collection="5110",
             dayOrNight="DNB"):
    '''
    Use SatPy to check if data exists already in root dir. If not contact the NASA LAADS DAC server to download the required data.
    '''

    # TODO: Is there away to get the filepath without two requests server? Doesn't seem to be be studying the API documentation

    assert 'LAADS_API_KEY' in os.environ
    query_response = requests.get(WEBSERVER_QUERY_URL + SEARCH_FOR_FILES, params={
        "collection": collection,
        "products": "VNP02IMG,VNP03IMG",
        "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "north": north,
        "south": south,
        "east": east,
        "west": west,
        "coordsOrTiles": "coords",
        "dayNightBoth": dayOrNight
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
    return download_files_from_server(root_dir, files), files
