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


if __name__ == "__main__":
    if input("Do you want to quarter the eye image and plot? (y/n)").lower() == 'y':
        path = input("Enter directory containing pickle files")
        with ProgressBar():
            pickle_paths = glob_pickle_files(path)
            for pickle in pickle_paths:
                ci = CycloneImage.load_cyclone_image(pickle)
                if ci.is_complete:
                    for y in range(-1, 1):
                        for x in range(-1, 1):
                            subimg = ci.new_rect(f"{x, y}", (ci.rmw / 2 + ci.rmw * y, ci.rmw / 2 + ci.rmw * x), ci.rmw,
                                                 ci.rmw)
                            ci.draw_rect(f"{x, y}")
    else:
        path = input("Enter pickle folder")
        pickle_paths = glob_pickle_files(path)
        points = []
        for path in pickle_paths:
            try:
                ci = CycloneImage.load_cyclone_image(path)
                subimg = ci.new_rect(f"da whole thing", (0, 0), ci.rmw * 2, ci.rmw * 2)
                gt, cat, basin, max_wind = ci.get_gt_and_intensity("da whole thing", mode="eyewall")
                if 200 < gt < 270:
                    points.append({"gt": gt, "cat": cat, "basin": basin, "max_wind": max_wind})
            except Exception as e:
                print(f"{path} --> error :{e}")

        plt.scatter((elem["gt"] for elem in points), (elem["max_wind"] for elem in points))
        plt.show()