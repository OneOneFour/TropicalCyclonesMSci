from AerosolImage import *


def test_MODIS_PATH():
    assert os.path.isdir(MODIS_PATH)

def test_MODIS_PATH_set():
    assert MODIS_PATH == os.getcwd()

def test_aerosol_image():
    checkpath = os.path.join(MODIS_PATH, str(2012), f"AerosolImage.001.gpz")
    assert AerosolImageMODIS.path(2012, 1) == checkpath

def test_aerosol_file():
    checkpath = os.path.join(MODIS_PATH, str(2012), f"new.001.c6.nc")
    assert AerosolImageMODIS.get_modis_file(2012,1) == checkpath

