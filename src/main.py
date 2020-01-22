import csv
import datetime

import matplotlib.pyplot as plt
import netCDF4 as nt
import numpy as np
from dask.diagnostics.progress import ProgressBar
import os

from CycloneImage import CycloneImage
from fetch_file import get_data
from CycloneSnapshot import CycloneSnapshot


def glob_pickle_files(directory):
    from glob import glob
    return glob(f"{directory}\*.pickle")


def pickle_file():
    fname = input("Enter file path of cyclone pickle")
    ci = CycloneSnapshot.load(fname)
    ci.draw_eye("I05")


if __name__ == "__main__":
    ci = pickle_file()