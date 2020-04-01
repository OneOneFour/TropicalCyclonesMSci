import numpy as np
from netCDF4 import Dataset

FILE = r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\simpleret_re_data.v1.nc"

RAD3770 = None
RE = None
SATZ = None
SOLZ = None
RAZ = None

with Dataset(FILE) as rootgrp:
    RE = np.array(rootgrp["re"])[::-1]
    SATZ = np.array(rootgrp["satz"])
    SOLZ = np.array(rootgrp["solz"])
    RAZ = np.array(rootgrp["raz"])
    RAD3770 = np.array(rootgrp["rad3770"])


def reduce(satz, solz, raz):
    solz_where = np.argmin(np.abs(SOLZ - solz))
    satz_where = np.argmin(np.abs(SATZ - satz))
    raz_where = np.argmin(np.abs(RAZ - raz))

    return RAD3770[:, solz_where, satz_where, raz_where]


def get_re(i4_r, satz, solz, raz, nan_out_of_range=False):
    re_lut = reduce(satz, solz, raz)[::-1]
    if nan_out_of_range:
        return np.interp(i4_r, re_lut, RE, left=np.nan, right=np.nan)
    return np.interp(i4_r, re_lut, RE)
