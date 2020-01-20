import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy.optimize as  sp

MIN_CUTOFF = -40
MAX_CUTOFF = 5

cubic = lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d
quadratic = lambda x, a, b, c: a * x ** 2 + b * x + c
d_gt_d_a = lambda a, b, c: (-3 * a * c + 2 * b * (-b + np.sqrt(b ** 2 - 3 * a * c))) / (
        6 * (a ** 2) * np.sqrt(b ** 2 - 3 * a * c))
d_gt_d_b = lambda a, b, c: (-1 + b / np.sqrt(b ** 2 - 3 * a * c)) / (3 * a)
d_gt_d_c = lambda a, b, c: -1 / (2 * np.sqrt(b ** 2 - 3 * a * c))


class GTFit:
    def __init__(self, i04_flat,i05_flat):
        self.i04 = i04_flat
        self.i05 = i05_flat
        self.gt = []



    def curve_fit_funcs(self, fitting_function=cubic):
        bbox = np.where(np.logical_and(self.i05 > MIN_CUTOFF, self.i05 < MAX_CUTOFF))
        x_i05 = self.i05[bbox]
        y_i04 = self.i04[bbox]

        (a, b, c, d), cov = sp.curve_fit(fitting_function, x_i05, y_i04, absolute_sigma=True)
        a_err, b_err, c_err, d_err = np.sqrt(np.diag(cov))

        gt_ve = (-b + np.sqrt(b ** 2 - 3 * a * c)) / (3 * a)
        # gt = -params[1] / (2 * params[0])
        # gt_err = np.sqrt((perr[1] / (2 * params[0])) ** 2 + (perr[0] * params[1] / (2 * params[0] ** 2)) ** 2)
        if np.iscomplex(gt_ve) or (min(x_i05) > gt_ve > max(x_i05)):
            return

        curve_fit_err = np.sqrt(((b_err*c)/(2*b*b))**2 + (c_err/2*b)**2)
        gt_err = curve_fit_err
        self.gt = [gt_ve, gt_err]
        return gt_ve, gt_err, (a, b, c, d)

    def curve_fit_modes(self, mode="median"):
        x_i05 = np.arange(MIN_CUTOFF, MAX_CUTOFF, 1)
        if len(x_i05) < 1:
            return
        y_i04 = np.array([0] * len(x_i05))
        num_vals_bins = []
        point_errs = np.array([0] * len(x_i05))
        for i, x in enumerate(x_i05):
            vals = self.i04[np.where(np.logical_and(self.i05 > (x - 0.5), self.i05 < (x + 0.5)))]
            if len(vals) < 1:
                continue
            if mode == "mean":
                y_i04[i] = np.mean(vals)
                mean_std = np.std(vals) / np.sqrt(len(vals))
                point_errs[i] = mean_std
            elif mode == "min":
                y_i04[i] = min(vals)
                point_errs[i] = 0.5 ** 2
            elif mode == "median":
                y_i04[i] = np.median(vals)
                median_std = 1.253 * np.std(vals) / np.sqrt(len(vals))
                point_errs[i] = median_std ** 2
            elif mode == "eyewall":
                if x_i05 < 235:             # Takes minimum values lower than theoretical min gt and v.v.
                    y_i04[i] = min(vals)
                else:
                    y_i04[i] = max(vals)
                point_errs[i] = 0.5 ** 2

            num_vals_bins.append(len(vals))  # list of number of I04 values that were in each I05 increment (for errors)

        zero_args = np.where(y_i04 == 0)
        x_i05 = np.delete(x_i05, zero_args)
        y_i04 = np.delete(y_i04, zero_args)
        point_errs = np.delete(point_errs, zero_args)

        params, cov = sp.curve_fit(cubic, x_i05, y_i04, absolute_sigma=True)
        perr = np.sqrt(np.diag(cov))

        xvalues = np.arange(min(x_i05), max(x_i05), 1)
        yvalues = cubic(xvalues, *params)
        gt_ve = (-params[1] + np.sqrt(params[1] ** 2 - 3 * params[0] * params[2])) / (3 * params[0])
        if np.iscomplex(gt_ve) or min(x_i05) > gt_ve > max(x_i05):
            return
        print("\n".join([f"{i}: Value: {param}, error:{perr[i]}" for i, param in enumerate(params)]))
        # satellite_data_error = ??
        curve_fit_err = np.sqrt(
            (d_gt_d_a(*params[:-1]) * perr[0]) ** 2 +
            (d_gt_d_b(*params[:-1]) * perr[1]) ** 2 +
            (d_gt_d_c(*params[:-1]) * perr[2]) ** 2
        )
        gt_err = curve_fit_err
        self.gt = [gt_ve, gt_err]

        return gt_ve, gt_err, params


