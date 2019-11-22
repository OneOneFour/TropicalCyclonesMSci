import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp


def curve_func(x, a, b, c, d):
    return a*x ** 3 + b*x**2 + c*x + d


def straight_line_func(x, m, b):
    return m*x + b


def plot():
    plt.scatter(i04flat, i05flat, s=0.25)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.show()


def gt_min_i04(i04flat, i05flat):
    idxs = np.nonzero(i05flat < 200)
    i04_fit_data = np.delete(i04flat, idxs[0])
    i05_fit_data = np.delete(i05flat, idxs[0])
    i04min_ind = np.unravel_index(np.argmin(i04_fit_data, axis=None), i04_fit_data.shape)
    gt = i05_fit_data[i04min_ind]
    return gt


def gt_curve_fit(i04flat, i05flat, mode="min", plot=False):
    idxs = np.nonzero(i05flat < 200)
    i04_fit_data = np.delete(i04flat, idxs[0])
    i05_fit_data = np.delete(i05flat, idxs[0])

    minimised_i05_fit_data = np.arange(int(min(i05_fit_data)), int(max(i05_fit_data)), 1)
    minimised_i04_fit_data = []

    for i in minimised_i05_fit_data:
        min_idxs = np.where(np.logical_and(i05_fit_data>(i-0.5), i05_fit_data<(i+0.5)))[0]
        i04_min_vals = []
        for idx in min_idxs:
            i04_min_vals.append(i04_fit_data[idx])
        if len(i04_min_vals) > 0:
            if mode == "mean":
                minimised_i04_fit_data.append(np.mean(i04_min_vals))
            elif mode == "min":
                minimised_i04_fit_data.append(min(i04_min_vals))
        else:
            minimised_i05_fit_data = np.delete(minimised_i05_fit_data, np.where(minimised_i05_fit_data == i)[0])

    params, cov = sp.curve_fit(curve_func, minimised_i05_fit_data, minimised_i04_fit_data)
    xvalues = np.arange(min(minimised_i05_fit_data), max(minimised_i05_fit_data), 1)
    yvalues = curve_func(xvalues, params[0], params[1], params[2], params[3])
    i04min_ind = np.argmin(yvalues)
    gt = xvalues[i04min_ind]

    if plot:
        plt.scatter(i05flat, i04flat, s=0.25)
        plt.plot(minimised_i05_fit_data, minimised_i04_fit_data)
        plt.plot(xvalues, yvalues)
        plt.show()

    return gt


def gt_two_line_fit(i04flat, i05flat, mode="min", plot=False):
    idxs = np.nonzero(i05flat < 200)
    i04_fit_data = np.delete(i04flat, idxs[0])
    i05_fit_data = np.delete(i05flat, idxs[0])

    minimised_i05_fit_data = np.arange(int(min(i05_fit_data)), int(max(i05_fit_data)), 1)
    minimised_i04_fit_data = []

    for i in minimised_i05_fit_data:
        min_idxs = np.where(np.logical_and(i05_fit_data > (i - 0.5), i05_fit_data < (i + 0.5)))[0]
        i04_min_vals = []
        for idx in min_idxs:
            i04_min_vals.append(i04_fit_data[idx])
        if len(i04_min_vals) > 0:
            if mode == "mean":
                minimised_i04_fit_data.append(np.mean(i04_min_vals))
            elif mode == "min":
                minimised_i04_fit_data.append(min(i04_min_vals))
        else:
            minimised_i05_fit_data = np.delete(minimised_i05_fit_data, np.where(minimised_i05_fit_data == i)[0])

    i05_lower_fit_data = minimised_i05_fit_data[:int(round(len(minimised_i05_fit_data)/2))]
    i05_upper_fit_data = minimised_i05_fit_data[int(round(len(minimised_i05_fit_data)/2)):]
    i04_lower_fit_data = minimised_i04_fit_data[:int(round(len(minimised_i04_fit_data)/2))]
    i04_upper_fit_data = minimised_i04_fit_data[int(round(len(minimised_i04_fit_data)/2)):]

    lower_params, lower_cov = sp.curve_fit(straight_line_func, i05_lower_fit_data, i04_lower_fit_data)
    upper_params, upper_cov = sp.curve_fit(straight_line_func, i05_upper_fit_data, i04_upper_fit_data)
    xvalues = np.arange(min(minimised_i05_fit_data), max(minimised_i05_fit_data), 1)
    lower_yvalues = straight_line_func(xvalues, lower_params[0], lower_params[1])
    upper_yvalues = straight_line_func(xvalues, upper_params[0], upper_params[1])
    intersection_idx = np.where(lower_yvalues<upper_yvalues)[0][0]
    gt = xvalues[intersection_idx]

    if plot:
        plt.scatter(i05flat, i04flat, s=0.25)
        plt.plot(i05_lower_fit_data, i04_lower_fit_data)
        plt.plot(i05_upper_fit_data, i04_upper_fit_data)
        plt.plot(xvalues[:intersection_idx], lower_yvalues[:intersection_idx])
        plt.plot(xvalues[intersection_idx:], upper_yvalues[intersection_idx:])
        plt.show()

    return gt


if __name__ == "__main__":

    dir = "proc/pic_dat_test/"
    filenames = [dir+"2013-10-19Cat4(2).pickle", dir+"2014-08-09Cat4(3).pickle", dir+"2014-10-16Cat4(4).pickle",
                 dir+"2014-11-04Cat4(3).pickle", dir+"2015-08-30Cat4(3).pickle"]

    for file_name in filenames:
        with open(file_name, "rb") as file:
            i04, i05 = pickle.load(file)

        gt_min = gt_min_i04(i04.flatten(), i05.flatten())
        gt_curve = gt_curve_fit(i04.flatten(), i05.flatten(), mode="min", plot=False)
        gt_straight = gt_two_line_fit(i04.flatten(), i05.flatten(), mode="min", plot=False)
        print("Curve gt: %f, Straight Line gt: %f, Min gt: %f" % (gt_curve, gt_straight, gt_min))
