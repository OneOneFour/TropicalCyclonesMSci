import numpy as np
import scipy.optimize as sp

cubic = lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d
quadratic = lambda x, a, b, c: a * x ** 2 + b * x + c
d_gt_d_a = lambda a, b, c: (-3 * a * c + 2 * b * (-b + np.sqrt(b ** 2 - 3 * a * c))) / (
        6 * (a ** 2) * np.sqrt(b ** 2 - 3 * a * c))
d_gt_d_b = lambda a, b, c: (-1 + b / np.sqrt(b ** 2 - 3 * a * c)) / (3 * a)
d_gt_d_c = lambda a, b, c: -1 / (2 * np.sqrt(b ** 2 - 3 * a * c))

HOMOGENEOUS_FREEZING_TEMP = -38


def piecewise_step(t, t_g, t_m, t_b, r_e_0, a, r_e_g):
    return np.piecewise(t, [t > t_b and t > t_m and t > t_g,
                            t_g < t and t_m < t and t < t_b,
                            t_m > t and t > t_g and t < t_b,
                            t < t_g and t < t_b and t < t_m],
                        [lambda t: a * t + r_e_0 - a * t_b, lambda t: r_e_0,
                         lambda t: (r_e_g - r_e_0) / (t_g - t_m) * t + r_e_0 - (r_e_g - r_e_0) / (t_g - t_m) * t_m,
                         lambda t: r_e_g])


def simple_piecewise(t, t_g, r_e, a, b):
    return np.piecewise(t, [t < t_g], [lambda t: np.abs(a) * (t_g - t) + r_e,
                                       lambda t: np.abs(b) * (t - t_g) + r_e])


class GTFit:
    def __init__(self, i04_flat, i05_flat):
        self.i04 = i04_flat
        self.i05 = i05_flat
        self.gt = []
        self.x_i05 = None
        self.y_i04 = None

    # def curve_fit_funcs(self, fitting_function=cubic):
    #     (a, b, c, d), cov = sp.curve_fit(fitting_function, self.i05, self.i04, absolute_sigma=True)
    #
    #     a_err, b_err, c_err, d_err = np.sqrt(np.diag(cov))
    #
    #     gt_ve = (-b + np.sqrt(b ** 2 - 3 * a * c)) / (3 * a)
    #     # gt = -params[1] / (2 * params[0])
    #     # gt_err = np.sqrt((perr[1] / (2 * params[0])) ** 2 + (perr[0] * params[1] / (2 * params[0] ** 2)) ** 2)
    #     if np.iscomplex(gt_ve) or (min(self.i05) > gt_ve > max(self.i05)):
    #         return
    #
    #     curve_fit_err = np.sqrt(((b_err * c) / (2 * b * b)) ** 2 + (c_err / 2 * b) ** 2)
    #     gt_err = curve_fit_err
    #     self.gt = [gt_ve, gt_err]
    #     return gt_ve, gt_err, (a, b, c, d)

    def piecewise_fit(self, fig=None, ax=None, func=simple_piecewise):
        self.x_i05 = self.i05
        self.y_i04 = self.i04
        if len(self.i05) or len(self.i04) < 4:
            raise ValueError("Problem underconstrained.")
        params, cov = sp.curve_fit(func, self.x_i05, self.y_i04, p0=(HOMOGENEOUS_FREEZING_TEMP, 220, 1, 1))
        gt_err = np.sqrt(np.diag(cov))[0]
        r2 = 1 - (np.sum((self.y_i04 - func(self.x_i05, *params)) ** 2)) / np.sum((self.y_i04 - self.y_i04.mean()) ** 2)

        self.gt = params[0]
        if fig and ax:
            self.plot(fig, ax, func=func, params=params)
        return (self.gt, gt_err), (r2, params)

    def piecewise_percentile(self, percentile=50, fig=None, ax=None):
        if len(self.i05) < 1:
            raise ValueError("I5 data is empty. This could be due to masking or a bad input")
        self.x_i05 = np.arange(min(self.i05), max(self.i05), 1)
        self.y_i04 = [0] * len(self.x_i05)
        if len(self.x_i05) < 1:
            raise ValueError("I4 data is empty. This could be due to masking or a bad input")
        for i, x in enumerate(self.x_i05):
            vals = self.i04[np.where(np.logical_and(self.i05 > (x - 0.5), self.i05 < (x + 0.5)))]
            if len(vals) == 0:
                continue
            self.y_i04[i] = np.percentile(vals, percentile)
        zero_args = np.where(self.y_i04 == 0)
        self.x_i05 = np.delete(self.x_i05, zero_args)
        self.y_i04 = np.delete(self.y_i04, zero_args)
        if len(self.x_i05) < 4:
            raise ValueError("Problem is under-constrained, less than 4 free parameters")

        params, cov = sp.curve_fit(simple_piecewise, self.x_i05, self.y_i04, absolute_sigma=True,
                                   p0=(HOMOGENEOUS_FREEZING_TEMP, 260, 1, 1))

        err = np.sqrt(np.diag(cov))
        self.gt_err = err[0]
        self.gt = params[0]
        r2 = 1 - (np.sum((self.y_i04 - simple_piecewise(self.x_i05, *params)) ** 2)) / np.sum(
            (self.y_i04 - self.y_i04.mean()) ** 2)
        if fig and ax:
            self.plot(fig, ax, func=simple_piecewise, params=params)
        return (self.gt, self.gt_err), (r2, params)

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
            ax.scatter(self.i04, self.i05, s=0.1)

        else:
            x = np.linspace(min(self.x_i05), max(self.x_i05))
            ax.scatter(self.y_i04, self.x_i05, s=0.1)

        if func:
            ax.plot([func(x_i, *params) for x_i in x], x, "y", label="Curve fit")
        ax.axhline(-38, xmin=min(x), xmax=max(x), lw=1, color="g")  # homogenous ice freezing temperature:
        ax.axhline(self.gt, xmin=min(x), xmax=max(x), lw=1, color="r")
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_ylabel("Cloud Top Temperature (C)")
        ax.set_xlabel("I4 band reflectance (K)")

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
