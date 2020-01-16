import csv
import datetime

import matplotlib.pyplot as plt
import netCDF4 as nt
import numpy as np
from dask.diagnostics.progress import ProgressBar

from CycloneImage import CycloneImage
from fetch_file import get_data



def glob_pickle_files(directory):
    from glob import glob
    return glob(f"{directory}\*.pickle")


def pickle_file():
    fname = input("Enter file path of cyclone pickle")
    with ProgressBar():
        ci = CycloneImage.load_cyclone_image(fname)
        ci.draw_eye("I05")
        ci.plot_derivatives()


if __name__ == "__main__":
    pickle_file()