import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import wget
import requests
import satpy


WEBSERVER_QUERY_URL = "http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices"

SEARCH_FOR_FILES = "/searchForFiles"
GET_FILE_URLS = "/getFileUrls"

#TODO find way to apply this filter after the first initial query, aka by looking inside the ncdump metadata
def get_data(root_dir, year, month, day, north=90, south=-90, west=-180, east=180, collection="5110",dayOrNight="DNB", mode="download"):
    '''
    Use SatPy to check if data exists already in root dir. If not contact the NASA LAADS DAC server to download the required data.
    '''

    # TODO: Is there away to get the filepath without two requests server? Doesn't seem to be be studying the API documentation

    assert 'LAADS_API_KEY' in os.environ
    y_o_day = datetime(year, month, day).timetuple().tm_yday
    query_response = requests.get(WEBSERVER_QUERY_URL + SEARCH_FOR_FILES, params={
        "collection": collection,
        "products": "VNP02IMG,VNP03IMG",
        "startTime": datetime(year, month, day).strftime("%Y-%m-%d"),
        "endTime": datetime(year, month, day).strftime("%Y-%m-%d"),
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
        fileIdStr += child.text + ","
    file_name_response = requests.get(WEBSERVER_QUERY_URL + GET_FILE_URLS, params={"fileIds": fileIdStr})
    files = [f.text for f in ET.fromstring(file_name_response.content)]
    fpath = [os.path.join(root_dir, os.path.split(f)[1]) for f in files]
    if mode == "best_track_compare":
        return fpath
    elif mode == "download":
        for i,file in enumerate(fpath):
            if os.path.isfile(file):
                print(f"{file} already downloaded... skipping...")
                continue
            print(f"Downloading {file}")
            # Begin downloading from the content server
            header = {
                "Authorization": os.environ["LAADS_API_KEY"]
            }
            # Use wget for large files
            # headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
            wget.download(files[i], file)
            if not os.path.isfile(file):
                raise ConnectionError(
                    f"File {files[i]} could not be downloaded to {file}. Please check your internet connection and try again!")
    return fpath
