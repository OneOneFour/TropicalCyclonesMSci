import glob
import os
from GTFit import GTFit
import numpy as np
import tqdm

from alt.CycloneMap import CycloneImageFast

ABSZ = 273.15
gt_eyewall = []
gt_error = []
files = glob.glob(os.path.join("D:\\cache", "**.gzp"))
for f in tqdm.tqdm(files):
    ci = CycloneImageFast.from_gzp(f)
    ci.eye().i4_reflectance *= 100
    try:
        gt,gt_i4,nrmse = GTFit(ci.eye().i4_reflectance,ci.eye().i5_flat).piecewise_fit()
    except AssertionError:
        continue

    if ABSZ - 40 < gt.value < ABSZ and nrmse <= 0.25:
        gt_eyewall.append(gt.value)
        gt_error.append(gt.error)

import matplotlib.pyplot as plt

plt.hist(gt_eyewall, bins=np.arange(ABSZ - 40, ABSZ - 10, 2))
plt.show()
