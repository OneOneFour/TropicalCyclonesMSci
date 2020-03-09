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
COLOR_LIST = ["firebrick", "gold", "darkorange"]


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

    def piecewise_fit(self, fig=None, ax=None, func=simple_piecewise, setup_axis=True):
        x_i05 = self.i05
        y_i04 = self.i04
        if len(self.i05) < 50 or len(self.i04) < 50:
            raise ValueError("Problem underconstrained.")
        if not self.i01 is None:
            params, cov = sp.curve_fit(func, x_i05, y_i04,
                                       p0=(HOMOGENEOUS_FREEZING_TEMP, 280, 1, 1),
                                       sigma=(100 / (self.i01 ** 2)))
        else:
            params, cov = sp.curve_fit(func, x_i05, y_i04,
                                       p0=(HOMOGENEOUS_FREEZING_TEMP, 280, 1, 1))
        r2 = 1 - (np.sum((y_i04 - func(x_i05, *params)) ** 2)) / np.sum((y_i04 - y_i04.mean()) ** 2)
        gt_err = np.sqrt(np.diag(cov))[0]
        gt = params[0]
        if fig and ax:
            self.plot(y_i04, x_i05, fig, ax, func=func, params=params, setup_axis=setup_axis)
        i4 = func(gt, *params)
        i4_err_appx = abs(gt_err * (i4 / gt))
        return GT(gt, gt_err), I4(i4, i4_err_appx), r2

    def piecewise_percentile_multiple(self, percentiles=None, fig=None, ax=None):
        self.plot(self.i04, self.i05, fig, ax, s=0.1, c='b')
        rtnLst = []
        for i, p in enumerate(percentiles):
            rtnLst.append(self.piecewise_percentile(percentile=p, fig=fig, ax=ax, setup_axis=False, c=COLOR_LIST[i]))
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_ylabel("Cloud Temperature (C)")
        ax.set_xlabel("I4 band reflectance (K)")
        ax.legend()

        return rtnLst

    def piecewise_percentile(self, percentile=50, fig=None, ax=None, setup_axis=True, c='r'):
        if len(self.i05) < 1:
            raise ValueError("I5 data is empty. This could be due to masking or a bad input")
        x_i05, y_i04 = self.bin_data(per_bin_func=np.percentile, bin_width=1, bin_func_args=(percentile,))

        params, cov = sp.curve_fit(simple_piecewise, x_i05, y_i04, absolute_sigma=True,
                                   p0=(HOMOGENEOUS_FREEZING_TEMP, 260, 1, 1))

        err = np.sqrt(np.diag(cov))
        gt_err = err[0]
        gt = params[0]
        r2 = 1 - (np.sum((y_i04 - simple_piecewise(x_i05, *params)) ** 2)) / np.sum(
            (y_i04 - y_i04.mean()) ** 2)
        if fig and ax:
            self.plot(y_i04, x_i05, fig, ax, gt, gt_err, func=simple_piecewise, params=params, setup_axis=setup_axis,
                      s=10, c=c,label=str(percentile)+"th")
        i4 = simple_piecewise(gt, *params)
        i4_err_appx = abs(gt_err * (i4 / gt))

        return GT(gt, gt_err), I4(i4, i4_err_appx), r2

    def plot(self, x, y, fig, ax, gt=None, gt_err=None, func=None, params=None, setup_axis=True, s=1.0, c='b',
             label=None):
        if fig is None or ax is None:
            return
        ax.scatter(x, y, s=s, c=c)

        if func:
            x_samples = np.linspace(min(y), max(y), 100)
            ax.plot([func(x_i, *params) for x_i in x_samples], x_samples, "y")
            ax.legend()
        if gt:
            ax.axhline(gt, label=f"$T_{{g,{label}}}$:{round(gt, 1)}±{round(gt_err, 1)} C", c=c)

        if setup_axis:
            ax.axhline(-38, c='g', label="$T_{g,homo}$")
