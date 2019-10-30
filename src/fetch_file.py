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
def get_data(root_dir, year, month, day, north=90, south=-90, west=-180, east=180, collection="5110",dayOrNight="DNB"):
    '''
    Use SatPy to check if data exists already in root dir. If not contact the NASA LAADS DAC server to download the required data.
    '''
    y_o_day = datetime(year, month, day).timetuple().tm_yday
    try:
        filenames = satpy.find_files_and_readers(start_time=datetime(year, month, day),
                                                 end_time=datetime(year, month, day) + timedelta(days=1),
                                                 reader="viirs_l1b",
                                                 base_dir=root_dir)
        return filenames
    except ValueError as e:
        assert 'LAADS_API_KEY' in os.environ
        print("Files not found in directory, attempting download...")
        # Query for file URLS
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
            "dayNightBoth":dayOrNight
        })
        fileIdStr = ""
        root_fids = ET.fromstring(query_response.content)
        for child in root_fids:
            fileIdStr += child.text + ","
        file_name_response = requests.get(WEBSERVER_QUERY_URL + GET_FILE_URLS, params={"fileIds": fileIdStr})
        files = [f.text for f in ET.fromstring(file_name_response.content)]
        fpath = [os.path.join(root_dir, os.path.split(f)[1]) for f in files]
        if input(f"The following files will be downloaded (y/n)?:\n{files}\n") == "y":
            for i, file in enumerate(files):
                if os.path.isfile(fpath[i]):
                    print(f"{fpath[i]} already downloaded... skipping...")
                    continue
                print(f"Downloading {file}")
                # Begin downloading from the content server
                header = {
                    "Authorization": os.environ["LAADS_API_KEY"]
                }
                # Use wget for large files
                wget.download(file, fpath[i])
                if not os.path.isfile(fpath[i]):
                    raise ConnectionError(
                        f"File {file} could not be downloaded to {fpath[i]}. Please check your internet connection and try again!")
            return fpath
        else:
            print("Aborting download")
            return None
