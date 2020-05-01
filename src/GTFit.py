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

HOMOGENEOUS_FREEZING_TEMP_C = -38
HOMOGENEOUS_FREEZING_TEMP_K = 273.15 - 38
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
        self.i04 = np.array(i04_flat)
        self.i05 = np.array(i05_flat)
        self.i04 = self.i04[~np.isnan(self.i04)]
        self.i05 = self.i05[~np.isnan(self.i05)]
        assert len(self.i04) == len(self.i05)

        self.i01 = i01_flat
        self.gt = None

    def bin_data(self, per_bin_func=lambda x: x, bin_width=1, bin_func_args=None, delete_zeros=True,
                 custom_range=None, secondary_function=None):
        if custom_range:
            assert len(custom_range) == 2
            x_i05 = np.arange(custom_range[0], custom_range[1], bin_width)
        else:
            x_i05 = np.arange(min(self.i05), max(self.i05), bin_width)
        y_i04 = np.array([0] * len(x_i05), dtype=np.float64)
        if secondary_function:
            y_i04_2 = np.array([0] * len(x_i05), dtype=np.float64)
        if len(x_i05) < 1:
            raise ValueError("I4 data is empty. This could be due to masking or a bad input")
        for i, x in enumerate(x_i05):
            vals = self.i04[np.where(np.logical_and(self.i05 > (x - 0.5), self.i05 < (x + 0.5)))]
            if len(vals) == 0:
                continue
            if bin_func_args:
                y_i04[i] = per_bin_func(vals, *bin_func_args)
                if secondary_function:
                    y_i04_2[i] = secondary_function(vals, *bin_func_args)
            else:
                y_i04[i] = per_bin_func(vals)
                if secondary_function:
                    y_i04_2[i] = secondary_function(vals)
        if delete_zeros:
            zero_args = np.where(y_i04 == 0)
            x_i05 = np.delete(x_i05, zero_args)
            y_i04 = np.delete(y_i04, zero_args)
            if secondary_function:
                y_i04_2 = np.delete(y_i04_2, zero_args)
        if secondary_function:
            return x_i05, y_i04, y_i04_2
        else:
            return x_i05, y_i04

    def piecewise_fit(self, fig=None, ax=None, func=simple_piecewise, setup_axis=True, units="kelvin", c='r', label="",
                      i4_units="kelvin",s=0.5):
        x_i05, y_i04 = self.bin_data(np.mean)
        trial = min(y_i04)
        trial_x = x_i05[np.argmin(y_i04)]
        assert len(self.i05) > 50
        if len(x_i05) < 5 or len(x_i05) < 5:
            raise AssertionError
        params, cov = sp.curve_fit(func, x_i05, y_i04,
                                   p0=(
                                       HOMOGENEOUS_FREEZING_TEMP_K,
                                       0,
                                       1,
                                       1),
                                   )

        gt_err = np.sqrt(np.diag(cov))[0]
        gt = params[0]
        sres = ((y_i04 - func(x_i05,*params))**2).sum()
        stot = ((y_i04 - np.mean(y_i04))**2).sum()
        rmse = np.sqrt(((y_i04 - func(x_i05, *params)) ** 2).sum() / (len(y_i04)))
        nrmse = rmse / (max(y_i04) - min(y_i04))
        if fig and ax:
            self.plot(y_i04, x_i05, fig, ax, gt, gt_err, func=simple_piecewise, params=params, setup_axis=setup_axis,
                      units=units, s=s, c=c, label="Fitted Profile", add_label=label)
        i4 = float(func(gt, *params))
        i4_err_appx = abs(gt_err * (i4 / gt))
        return GT(gt, gt_err), I4(i4, i4_err_appx),nrmse

    def piecewise_percentile_multiple(self, percentiles=None, units="kelvin", fig=None, ax=None, plot_points=True,
                                      setup_axis=True,
                                      colors=None, label="", i4_units="kelvin"):
        if colors is None:
            colors = COLOR_LIST
        if plot_points:
            self.plot(self.i04, self.i05, fig, ax, s=0.5, c='g', setup_axis=False)
        rtnLst = []
        for i, p in enumerate(percentiles):
            rtnLst.append(
                self.piecewise_percentile(percentile=p, fig=fig, ax=ax, setup_axis=False, c=colors[i], label=label,
                                          units=units, i4_units=i4_units))
        if ax is not None and setup_axis:
            if units == "celcius":
                ax.axhline(-38, c='g', label="$T_{g,homo}$", ls="--")
            elif units == "kelvin":
                ax.axhline(273.15-38, c="g", label="$T_{g,homo}$", ls="--")
            ax.invert_yaxis()
            ax.invert_xaxis()
            ax.set_ylabel("Cloud Temperature (C)")
            ax.set_xlabel("I4 band reflectance (K)")
            ax.legend()
        return rtnLst

    def piecewise_percentile(self, percentile=50, fig=None, ax=None, setup_axis=True, units="kelvin", c='r', label="",
                             i4_units="kelvin"):
        if len(self.i05) < 1:
            raise ValueError("I5 data is empty. This could be due to masking or a bad input")
        x_i05, y_i04 = self.bin_data(per_bin_func=np.percentile, bin_width=1, bin_func_args=(percentile,))
        trial = min(y_i04)
        trial_x = x_i05[np.argmin(y_i04)]
        if len(x_i05) < 5 or len(y_i04) < 5:
            raise ValueError("Problem underconstrained")
        params, cov = sp.curve_fit(simple_piecewise, x_i05, y_i04, absolute_sigma=True,
                                   p0=(
                                       trial_x,
                                       trial, 1, 1))

        err = np.sqrt(np.diag(cov))
        gt_err = err[0]
        gt = params[0]
        rmse = np.sqrt(((y_i04 - simple_piecewise(x_i05, *params)) ** 2).sum() / (len(y_i04)))
        nrmse = rmse / (max(y_i04) - min(y_i04))
        if fig and ax:
            self.plot(y_i04, x_i05, fig, ax, gt, gt_err, func=simple_piecewise, params=params, setup_axis=setup_axis,
                      units=units, s=0.05, c=c, label=str(percentile) + "th", add_label=label)
        i4 = float(simple_piecewise(gt, *params))
        i4_err_appx = abs(gt_err * (i4 / gt))

        return GT(gt, gt_err), I4(i4, i4_err_appx), nrmse

    def plot(self, x, y, fig, ax, gt=None, gt_err=None, func=None, params=None, setup_axis=True, units="kelvin",
             s=0.5, c='b',
             label="None", add_label=""):
        if fig is None or ax is None:
            return
        ax.scatter(x, y, s=s, c="black")

        if func:
            x_samples = np.linspace(min(y), max(y), 100)
            ax.plot([func(x_i, *params) for x_i in x_samples], x_samples, c=c, label=f"{add_label} {label}")
            ax.legend()
        if gt:
            ax.axhline(gt, label=f"{add_label} $T_{{g,fitted}}$:{round(gt, 1)}±{round(gt_err, 1)} C", c=c, ls="--")

        if setup_axis:
            if units == "celcius":
                ax.axhline(-38, c='g', label="$T_{g,homo}$")
            elif units == "kelvin":
                ax.axhline(273.15 - 38, c="g", label="$T_{g,homo}$")
            ax.invert_yaxis()
            ax.invert_xaxis()
            ax.set_ylabel("Temperature (K)")
            ax.set_xlabel("I4 band reflectance (%)")
            ax.legend()
