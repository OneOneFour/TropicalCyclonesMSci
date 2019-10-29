import os
import xml.etree.ElementTree as ET
from datetime import datetime
import wget
import requests
import satpy

WEBSERVER_QUERY_URL = "http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices"

SEARCH_FOR_FILES = "/searchForFiles"
GET_FILE_URLS = "/getFileUrls"


def get_data(root_dir, year, month, day, north=90, south=-90, west=-180, east=180, collection="5100"):
    '''
    Use SatPy to check if data exists already in root dir. If not contact the NASA LAADS DAC server to download the required data.
    '''
    y_o_day = datetime(year, month, day).timetuple().tm_yday
    try:
        filenames = satpy.find_files_and_readers(start_time=datetime(year, month, day, 0, 0, 0),
                                                 end_time=datetime(year, month, day, 23, 59, 59), reader="viirs_1lb",
                                                 base_dir=root_dir)
    except ValueError:
        assert 'LAADS_API_KEY' in os.environ
        print("Files not found in directory, attempting download...")
        # Query for file URLS
        query_response = requests.get(WEBSERVER_QUERY_URL + SEARCH_FOR_FILES, params={
            "collection": collection,
            "products": "VNP02IMG,VNP03IMG",
            "startTime": datetime(year, month, day).strftime("%Y-%M-%D"),
            "endTime": datetime(year, month, day).strftime("%Y-%M-%D"),
            "north": north,
            "south": south,
            "east": east,
            "west": west,
            "coordsOrTiles": "coords",
        })
        fileIdStr = ""
        root_fids = ET.fromstring(query_response.content)
        for child in root_fids:
            fileIdStr += child.text + ","
        file_name_response = requests.get(WEBSERVER_QUERY_URL + GET_FILE_URLS, params={"fileIds": fileIdStr})
        files = [f.text for f in ET.fromstring(file_name_response.content)]
        if input(f"The following files will be downloaded (y/n)?:\n{files}") == "y":
            for file in files:
                print(f"Downloading {file}")
                # Begin downloading from the content server
                header = {
                    "Authorization": os.environ["LAADS_API_KEY"]
                }
                wget.download(file)
        else:
            print("Aborting download")
            return None



