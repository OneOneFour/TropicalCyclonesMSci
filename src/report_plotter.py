import matplotlib.pyplot as plt
import numpy as np
from alt.CycloneMap import CycloneImageFast

cyclone = "DORIAN.264315.gzp"
EYEWALL_DEMO = (f"D:\cache_m9test\{cyclone}")
EYEWALL_2_DEMO = f"D:\cache\{cyclone}"


def get_grid(cif: CycloneImageFast):
    cif.draw_grid()


if __name__ == "__main__":
    plt.ioff()
    cif_2 = CycloneImageFast.from_gzp(EYEWALL_DEMO)
    cif_2.eye().plot_raw_profile()
    cif_2.eye().plot("BTD_m")
    cif_2.eye().plot("i5")
    # get_grid(cif)
