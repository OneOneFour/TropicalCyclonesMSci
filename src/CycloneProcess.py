import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from GTFit import GTFit
from alt.CycloneMap import CycloneImageFast

CACHE_DIRECTORY = os.environ["CACHE_DIRECTORY"]

if __name__ == "__main__":
    files = glob.glob(os.path.join(CACHE_DIRECTORY, "**.gzp"))
    percentiles = (5, 50, 95)
    eye_gts = []
    enviro_gts = []
    eyes_i4, eyes_i5 = [[] for p in percentiles], [[] for p in percentiles]
    environs_i4, environs_i5 = [[] for p in percentiles], [[] for p in percentiles]
    for f in files:
        try:
            ci = CycloneImageFast.from_gzp(f)
            # EYE
            if ci.eye().good_gt:
                eye_i4, eye_i5 = ci.eye().bin_data_percentiles(percentiles)
                eye_gts.append(ci.eye().gt.value)
            environ_i4, environ_i5 = ci.environment_bin_percentiles(percentiles)
            enviro_gts.extend([c.gt.value for c in ci.cells])
            for i in range(len(percentiles)):
                environs_i4[i].extend(environ_i4[i])
                environs_i5[i].extend(environ_i5[i])
                try:
                    eyes_i4[i].extend(eye_i4[i])
                    eyes_i5[i].extend(eye_i5[i])
                except NameError:
                    continue
        except AssertionError:
            continue

    fig, ax = plt.subplots( figsize=(8, 8))
    i4_t = np.linspace(310,260,1)
    for i, p in enumerate(percentiles):
        gt_fit_eye = GTFit(eyes_i4[i], eyes_i5[i])
        i5, i4 = gt_fit_eye.bin_data(np.mean, 1)
        ax.plot(i4, i5, label=f"{p}th percentile eye")
        enviro_fit_eye = GTFit(environs_i4[i], environs_i5[i])
        i5_en, i4_en = enviro_fit_eye.bin_data(np.mean, 1)

        ax.plot(i4_en, i5_en, label=f"{p}th percentile external ")
    ax.plot(i4_t,[i4_ti - 273.15 for i4_ti in i4_t])
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("I4 band reflectance(K)")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    plt.show()