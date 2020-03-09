from collections import namedtuple

import numpy as np
import scipy.optimize as sp

GT = namedtuple("GT", ("value", "error"))
I4 = namedtuple("I4", ("value", "error"))

cubic = lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d
quadratic = lambda x, a, b, c: a * x ** 2 + b * x + c
d_gt_d_a = lambda a, b, c: (-3 * a * c + 2 * b * (-b + np.sqrt(b ** 2 - 3 * a * c))) / (
        6 * (a ** 2) * np.sqrt(b ** 2 - 3 * a * c))
d_gt_d_b = lambda a, b, c: (-1 + b / np.sqrt(b ** 2 - 3 * a * c)) / (3 * a)
d_gt_d_c = lambda a, b, c: -1 / (2 * np.sqrt(b ** 2 - 3 * a * c))

HOMOGENEOUS_FREEZING_TEMP = -38


def piecewise_step(t, t_g, t_m, t_b, r_e_0, a, r_e_g):
    return np.piecewise(t, [(t > t_b) & (t > t_m) & (t > t_g),
                            (t_g < t) & (t_m < t) & (t < t_b),
                            (t_m > t) & (t > t_g) & (t < t_b),
                            (t < t_g) & (t < t_b) & (t < t_m)],
                        [lambda t: a * t + r_e_0 - a * t_b, lambda t: r_e_0,
                         lambda t: (r_e_g - r_e_0) / (t_g - t_m) * t + r_e_0 - (r_e_g - r_e_0) / (t_g - t_m) * t_m,
                         lambda t: r_e_g])


def simple_piecewise(t, t_g, r_e, a, b):
    return np.piecewise(t, [t < t_g], [lambda t: np.abs(a) * (t_g - t) + r_e,
                                       lambda t: np.abs(b) * (t - t_g) + r_e])


class GTFit:
    def __init__(self, i04_flat, i05_flat, i01_flat=None):
        self.i04 = i04_flat
        self.i05 = i05_flat
        self.i01 = i01_flat
        self.gt = None
        self.x_i05 = None
        self.y_i04 = None

    def bin_data(self, per_bin_func=lambda x, a: x, bin_width=1, bin_func_args=None, delete_zeros=True,
                 custom_range=None):
        if custom_range:
            assert len(custom_range) == 2
            x_i05 = np.arange(custom_range[0], custom_range[1], bin_width)
        else:
            x_i05 = np.arange(min(self.i05), max(self.i05), bin_width)
        y_i04 = np.array([0] * len(x_i05))
        if len(x_i05) < 1:
            raise ValueError("I4 data is empty. This could be due to masking or a bad input")
        for i, x in enumerate(x_i05):
            vals = self.i04[np.where(np.logical_and(self.i05 > (x - 0.5), self.i05 < (x + 0.5)))]
            if len(vals) == 0:
                continue
            y_i04[i] = per_bin_func(vals, *bin_func_args)
        if delete_zeros:
            zero_args = np.where(y_i04 == 0)
            x_i05 = np.delete(x_i05, zero_args)
            y_i04 = np.delete(y_i04, zero_args)
        return x_i05, y_i04

    def piecewise_fit(self, fig=None, ax=None, func=simple_piecewise):
        self.x_i05 = self.i05
        self.y_i04 = self.i04
        if len(self.i05) < 50 or len(self.i04) < 50:
            raise ValueError("Problem underconstrained.")
        if not self.i01 is None:
            params, cov = sp.curve_fit(func, self.x_i05, self.y_i04,
                                       p0=(HOMOGENEOUS_FREEZING_TEMP, 280, 1, 1),
                                       sigma=(100 / (self.i01 ** 2)))
        else:
            params, cov = sp.curve_fit(func, self.x_i05, self.y_i04,
                                       p0=(HOMOGENEOUS_FREEZING_TEMP, 280, 1, 1))
        r2 = 1 - (np.sum((self.y_i04 - func(self.x_i05, *params)) ** 2)) / np.sum((self.y_i04 - self.y_i04.mean()) ** 2)
        gt_err = np.sqrt(np.diag(cov))[0]
        self.gt = params[0]
        if fig and ax:
            self.plot(fig, ax, func=func, params=params)
        i4 = func(self.gt, *params)
        i4_err_appx = abs(gt_err * (i4 / self.gt))
        return GT(self.gt, gt_err), I4(i4, i4_err_appx), r2

    def piecewise_percentile(self, percentile=50, fig=None, ax=None):
        if len(self.i05) < 1:
            raise ValueError("I5 data is empty. This could be due to masking or a bad input")
        self.x_i05, self.y_i04 = self.bin_data(per_bin_func=np.percentile, bin_width=1, bin_func_args=(percentile,))

        params, cov = sp.curve_fit(simple_piecewise, self.x_i05, self.y_i04, absolute_sigma=True,
                                   p0=(HOMOGENEOUS_FREEZING_TEMP, 260, 1, 1))

        err = np.sqrt(np.diag(cov))
        self.gt_err = err[0]
        self.gt = params[0]
        r2 = 1 - (np.sum((self.y_i04 - simple_piecewise(self.x_i05, *params)) ** 2)) / np.sum(
            (self.y_i04 - self.y_i04.mean()) ** 2)
        if fig and ax:
            self.plot(fig, ax, func=simple_piecewise, params=params)  #
        i4 = simple_piecewise(self.gt, *params)
        i4_err_appx = abs(self.gt_err * (i4 / self.gt))

        return GT(self.gt, self.gt_err), I4(i4, i4_err_appx), r2

    def curve_fit_percentile(self, percentile=50, fig=None, ax=None):
        self.x_i05 = np.arange(min(self.i05), max(self.i05), 1)
        self.y_i04 = [0] * len(self.x_i05)
        if len(self.x_i05) < 1:
            return
        for i, x in enumerate(self.x_i05):
            vals = self.i04[np.where(np.logical_and(self.i05 > (x - 0.5), self.i05 < (x + 0.5)))]
            if len(vals) == 0:
                continue
            self.y_i04[i] = np.percentile(vals, percentile)
        zero_args = np.where(self.y_i04 == 0)
        self.x_i05 = np.delete(self.x_i05, zero_args)
        self.y_i04 = np.delete(self.y_i04, zero_args)

        params, cov = sp.curve_fit(cubic, self.x_i05, self.y_i04, absolute_sigma=True)
        perr = np.sqrt(np.diag(cov))

        gt_ve = (-params[1] + np.sqrt(params[1] ** 2 - 3 * params[0] * params[2])) / (3 * params[0])
        if np.iscomplex(gt_ve) or min(self.x_i05) > gt_ve > max(self.x_i05):
            return
        # satellite_data_error = ??
        curve_fit_err = np.sqrt(
            (d_gt_d_a(*params[:-1]) * perr[0]) ** 2 +
            (d_gt_d_b(*params[:-1]) * perr[1]) ** 2 +
            (d_gt_d_c(*params[:-1]) * perr[2]) ** 2
        )
        gt_err = curve_fit_err
        self.gt = gt_ve

        if fig and ax:
            self.plot(fig, ax, func=cubic, params=params)
        return gt_ve, gt_err, params

    def plot(self, fig, ax, func=None, params=None):
        if fig is None or ax is None:
            return
        if self.x_i05 is None:
            x = np.linspace(min(self.i05), max(self.i05))
            ax.scatter(self.i04, self.i05, s=10, label="Fitted Points")

        else:
            x = np.linspace(min(self.x_i05), max(self.x_i05))
            ax.scatter(self.y_i04, self.x_i05, s=10, label="Fitted Points")

        if func:
            ax.plot([func(x_i, *params) for x_i in x], x, "y", label="Line of Best Fit")
            ax.legend()

        ax.axhline(self.gt)
        ax.axhline(self.gt + self.gt_err, c='b', linestyle='--')
        ax.axhline(self.gt - self.gt_err, c='b', linestyle='--')
        ax.axhline(-38)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_ylabel("Cloud Temperature (C)")
        ax.set_xlabel("I4 band reflectance (K)")
        ax.legend()

    def curve_fit_fraction_mean(self, low=0, high=1, fig=None, ax=None):
        """

        :param low: nth percentile to start from
        :param high: nth percentile to go to
        :return: Glaciation temperature associated error and fitting parameters
        """
        self.x_i05 = np.arange(min(self.i05), max(self.i05), 1)
        self.y_i04 = [0] * len(self.x_i05)
        if len(self.x_i05) < 1:
            return
        num_vals_bin = []
        point_errs = np.array([0] * len(self.x_i05))
        for i, x in enumerate(self.x_i05):
            vals = self.i04[np.where(np.logical_and(self.i05 > (x - 0.5), self.i05 < (x + 0.5)))].flatten()
            self.y_i04[i] = vals[int(low * len(vals)):int(high * len(vals))].mean()
            point_errs[i] = vals[int(low * len(vals)):int(high * len(vals))].std()

        zero_args = np.where(self.y_i04 == 0)
        self.x_i05 = np.delete(self.x_i05, zero_args)
        self.y_i04 = np.delete(self.y_i04, zero_args)
        point_errs = np.delete(point_errs, zero_args)

        params, cov = sp.curve_fit(cubic, self.x_i05, self.y_i04, absolute_sigma=True)
        perr = np.sqrt(np.diag(cov))

        gt_ve = (-params[1] + np.sqrt(params[1] ** 2 - 3 * params[0] * params[2])) / (3 * params[0])
        if np.iscomplex(gt_ve) or min(self.x_i05) > gt_ve > max(self.x_i05):
            return
        # satellite_data_error = ??
        curve_fit_err = np.sqrt(
            (d_gt_d_a(*params[:-1]) * perr[0]) ** 2 +
            (d_gt_d_b(*params[:-1]) * perr[1]) ** 2 +
            (d_gt_d_c(*params[:-1]) * perr[2]) ** 2
        )
        gt_err = curve_fit_err
        self.gt = gt_ve
        self.plot(fig, ax, func=cubic, params=params)
        return gt_ve, gt_err, params

    # TODO: Move binning into standard function
    # TODO: Using common x,y is kinda bad
    def plot_binned(self, fig, ax, bin_width=1):
        self.x_i05, self.y_i04 = self.bin_data(per_bin_func=np.mean, )

        self.plot(fig, ax)

    def curve_fit_modes(self, mode="median", fig=None, ax=None):
        x_i05 = np.arange(min(self.i05), max(self.i05), 1)
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
                vals_5_min = []
                vals_5_min_i05val = []
                if len(vals) < 1:
                    continue
                if x > 235:  # Takes minimum values lower than theoretical min gt and v.v.
                    for j in range(int(np.ceil(len(vals) / 20))):
                        if len(vals) == 0:  # Not all values of i05 will have 5 i04 values
                            break
                        vals_5_min.append(min(vals))
                else:
                    percent_range = int(np.ceil(len(vals) / 20))
                    for j in range(percent_range):
                        if len(vals) == 0:  # Not all values of i05 will have 5 i04 values
                            break
                        vals.sort()
                        idx = int((235 - x) * 0.025 * len(vals) + j)
                        vals_5_min.append(vals[idx])

                y_i04[i] = np.median(vals_5_min)
                point_errs[i] = 0.5 ** 2  # TODO: Fix errors

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
        # satellite_data_error = ??
        curve_fit_err = np.sqrt(
            (d_gt_d_a(*params[:-1]) * perr[0]) ** 2 +
            (d_gt_d_b(*params[:-1]) * perr[1]) ** 2 +
            (d_gt_d_c(*params[:-1]) * perr[2]) ** 2
        )
        gt_err = curve_fit_err
        self.gt = gt_ve
        self.plot(fig, ax, func=cubic, params=params)
        return gt_ve, gt_err, params

    def gt_via_minimum(self, fig=None, ax=None):
        min_arg = np.argmin(self.i04)
        self.gt = self.i05[min_arg]
        self.x_i05 = self.i05
        self.y_i04 = self.i04
        self.plot(fig, ax)
        return self.gt

    def gt_via_minimum_percentile(self, percentile, fig=None, ax=None):
        self.x_i05 = np.arange(min(self.i05), max(self.i05), 1)
        self.y_i04 = [0] * len(self.x_i05)
        if len(self.x_i05) < 1:
            return
        for i, x in enumerate(self.x_i05):
            vals = self.i04[np.where(np.logical_and(self.i05 > (x - 0.5), self.i05 < (x + 0.5)))]
            self.y_i04[i] = np.percentile(vals, percentile)
        min_arg = np.argmin(self.y_i04)
        self.gt = self.x_i05[min_arg]
        self.plot(fig, ax)
        return self.gt
