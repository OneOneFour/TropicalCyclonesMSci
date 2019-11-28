import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp


def curve_func(x, a, b, c, d):
    return a*x ** 3 + b*x**2 + c*x + d


def straight_line_func(x, m, b):
    return m*x + b


def gt_min_i04(i04flat, i05flat):
    idxs = np.nonzero(i05flat < 200)
    i04_fit_data = np.delete(i04flat, idxs[0])
    i05_fit_data = np.delete(i05flat, idxs[0])
    i04min_ind = np.unravel_index(np.argmin(i04_fit_data, axis=None), i04_fit_data.shape)
    gt = i05_fit_data[i04min_ind]
    gt_err = np.std(i04flat)
    return gt, gt_err


def gt_curve_fit(i04flat, i05flat, mode="min", plot=False):
    idxs = np.nonzero(i05flat < 210)
    i04_fit_data = np.delete(i04flat, idxs[0])
    i05_fit_data = np.delete(i05flat, idxs[0])

    minimised_i05_fit_data = np.arange(int(min(i05_fit_data)), int(max(i05_fit_data)), 1)
    minimised_i04_fit_data = []
    num_vals_bins = []
    point_errs = 0              # Errors due to variance in I04 temps (y-data)

    for i in minimised_i05_fit_data:
        min_idxs = np.where(np.logical_and(i05_fit_data>(i-0.5), i05_fit_data<(i+0.5)))[0]
        i04_min_vals = []
        for idx in min_idxs:
            i04_min_vals.append(i04_fit_data[idx])
        if len(i04_min_vals) > 0:
            if mode == "mean":
                minimised_i04_fit_data.append(np.mean(i04_min_vals))
                mean_std = np.std(i04_min_vals)/np.sqrt(len(i04_min_vals))
                point_errs += mean_std ** 2
            elif mode == "min":
                minimised_i04_fit_data.append(min(i04_min_vals))
                point_errs += 0.5 ** 2
            elif mode == "median":
                median_i04 = np.median(i04_min_vals)
                minimised_i04_fit_data.append(median_i04)
                median_std = 1.253 * np.std(i04_min_vals)/np.sqrt(len(i04_min_vals))
                point_errs += median_std ** 2
        else:
            minimised_i05_fit_data = np.delete(minimised_i05_fit_data, np.where(minimised_i05_fit_data == i)[0])

        num_vals_bins.append(len(i04_min_vals))         # list of number of I04 values that were in each I05 increment (for errors)

    params, cov = sp.curve_fit(curve_func, minimised_i05_fit_data, minimised_i04_fit_data)
    # perr = np.sqrt(np.diag(cov))

    xvalues = np.arange(min(minimised_i05_fit_data), max(minimised_i05_fit_data), 1)
    yvalues = curve_func(xvalues, params[0], params[1], params[2], params[3])
    i04min_ind = np.argmin(yvalues)
    gt = xvalues[i04min_ind]

    # satellite_data_error = ??
    curve_fit_err = 0.5 * abs(3*params[0]*gt**2 + 2*params[1]*gt + params[2])
    gt_err = np.sqrt(curve_fit_err ** 2 + 0.5 ** 2 + point_errs)

    if plot:
        plt.scatter(i05flat, i04flat, s=0.25)
        plt.plot(minimised_i05_fit_data, minimised_i04_fit_data, label="Data to be fitted")
        plt.plot(xvalues, yvalues, label="Line of best fit")
        plt.legend()
        plt.show()

    return gt, gt_err


def gt_two_line_fit(i04flat, i05flat, mode="min", plot=False):
    idxs = np.nonzero(i05flat < 210)
    i04_fit_data = np.delete(i04flat, idxs[0])
    i05_fit_data = np.delete(i05flat, idxs[0])

    minimised_i05_fit_data = np.arange(int(min(i05_fit_data)), int(max(i05_fit_data)), 1)
    minimised_i04_fit_data = []
    num_vals = 0
    point_errs = 0
    found_middle = False

    for i in minimised_i05_fit_data:
        min_idxs = np.where(np.logical_and(i05_fit_data > (i - 0.5), i05_fit_data < (i + 0.5)))[0]
        i04_min_vals = []
        for idx in min_idxs:
            i04_min_vals.append(i04_fit_data[idx])

        if len(i04_min_vals) > 0:
            num_vals += len(i04_min_vals)
            if mode == "mean":
                minimised_i04_fit_data.append(np.mean(i04_min_vals))
                mean_std = np.std(i04_min_vals) / np.sqrt(len(i04_min_vals))
                point_errs += mean_std ** 2
            elif mode == "min":
                minimised_i04_fit_data.append(min(i04_min_vals))
                point_errs += 0.5 ** 2
            elif mode == "median":
                median_i04 = np.median(i04_min_vals)
                minimised_i04_fit_data.append(median_i04)
                median_std = 1.253 * np.std(i04_min_vals) / np.sqrt(len(i04_min_vals))
                point_errs += median_std ** 2
        else:
            minimised_i05_fit_data = np.delete(minimised_i05_fit_data, np.where(minimised_i05_fit_data == i)[0])

        if not found_middle:
            if num_vals >= len(i04_fit_data)/2:
                mid_point = np.where(minimised_i05_fit_data==i)[0]
                found_middle = True

    for i in mid_point:
        i05_lower_fit_data = minimised_i05_fit_data[:i]
        i05_upper_fit_data = minimised_i05_fit_data[i:]
        i04_lower_fit_data = minimised_i04_fit_data[:i]
        i04_upper_fit_data = minimised_i04_fit_data[i:]

        lower_params, lower_cov = sp.curve_fit(straight_line_func, i05_lower_fit_data, i04_lower_fit_data)
        upper_params, upper_cov = sp.curve_fit(straight_line_func, i05_upper_fit_data, i04_upper_fit_data)
        xvalues = np.arange(min(minimised_i05_fit_data), max(minimised_i05_fit_data), 1)
        lower_yvalues = straight_line_func(xvalues, lower_params[0], lower_params[1])
        upper_yvalues = straight_line_func(xvalues, upper_params[0], upper_params[1])

        fit_err_lower = abs(lower_params[0]) * 0.5
        fit_err_upper = abs(upper_params[0]) * 0.5
        fit_err_combined = np.sqrt(fit_err_upper**2 + fit_err_lower**2)
        gt_err = np.sqrt(fit_err_combined**2 + 2*0.5**2 + point_errs)

        try:
            intersection_idx = np.where(lower_yvalues<upper_yvalues)[0][0]
            gt = xvalues[intersection_idx]

            if plot:
                plt.scatter(i05flat, i04flat, s=0.25)
                plt.plot(i05_lower_fit_data, i04_lower_fit_data)
                plt.plot(i05_upper_fit_data, i04_upper_fit_data)
                plt.plot(xvalues[:intersection_idx], lower_yvalues[:intersection_idx])
                plt.plot(xvalues[intersection_idx:], upper_yvalues[intersection_idx:])
                plt.show()
        except IndexError:
            print("No min value")
            gt = 0

    return gt, gt_err


if __name__ == "__main__":

    dir = "proc/pic_dat_test/"
    filenames = [dir+"2013-10-19Cat4(2).pickle", dir+"2014-08-09Cat4(3).pickle", dir+"2014-10-16Cat4(4).pickle",
                 dir+"2014-11-04Cat4(3).pickle", dir+"2015-08-30Cat4(3).pickle"]

    for file_name in filenames:
        with open(file_name, "rb") as file:
            i04, i05 = pickle.load(file)

        gt_min, gt_min_err = gt_min_i04(i04.flatten(), i05.flatten())
        gt_curve, gt_curve_err = gt_curve_fit(i04.flatten(), i05.flatten(), mode="mean", plot=False)
        gt_straight, gt_straight_err = gt_two_line_fit(i04.flatten(), i05.flatten(), mode="mean", plot=False)
        curve_min_diff = abs(gt_curve - gt_min) / gt_min * 100
        straight_min_diff = abs(gt_straight - gt_min) / gt_min * 100
        curve_straight_diff = abs(gt_curve - gt_straight) / gt_straight * 100
        print("Curve gt: %i+-%f, Straight Line gt: %i+-%f, Min gt: %f+-%f"
              % (gt_curve, gt_curve_err, gt_straight, gt_straight_err, gt_min, gt_min_err))
        print("Curve-Straight Diff = %f, Curve-Min diff = %f, Straight-Min diff = %f"
              % (curve_straight_diff, curve_min_diff, straight_min_diff))
