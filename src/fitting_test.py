import pickle
import numpy as np
import matplotlib.pyplot as plt

file_name = "proc/pic_dat_test/2013-10-19Cat4(2).pickle"
with open(file_name, "rb") as file:
    i04, i05 = pickle.load(file)

i04flat = i04.flatten()
i05flat = i05.flatten()

def plot():
    plt.scatter(i04flat, i05flat, s=0.25)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.show()

def gt_min_i04():
    i04min = np.amin(i04flat)
    i04min_ind = np.unravel_index(np.argmin(i04flat, axis=None), i04flat.shape)
    gt = i05flat[i04min_ind]
    print(gt)

def gt_curve_fit():


def gt_two_line_fit():


gt_min_i04()
plot()