import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy.optimize as  sp

MIN_CUTOFF = 210

cubic = lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d


class SubImage:
    def __init__(self, i04, i05, w, h, center):
        self.__i04 = i04
        self.__i05 = i05
        self.__w = w
        self.__h = h
        self.__center = center

    @property
    def i04(self):
        return self.__i04

    @property
    def i04_flat(self):
        return self.__i04.flatten()

    @property
    def i05_flat(self):
        return self.__i05.flatten()

    @property
    def i05(self):
        return self.__i05

    @property
    def width(self):
        return self.__w

    @property
    def height(self):
        return self.__h

    @property
    def center(self):
        return self.__center

    def curve_fit(self, mode="mean", plot=False):
        x_i05 = np.arange(MIN_CUTOFF, int(max(self.i05_flat)), 1)
        if len(x_i05) < 1:
            return
        y_i04 = np.array([0] * len(x_i05))
        num_vals_bins = []
        point_errs = np.array([0] * len(x_i05))
        for i, x in enumerate(x_i05):
            vals = self.i04_flat[np.where(np.logical_and(self.i05_flat > (x - 0.5), self.i05_flat < (x + 0.5)))]
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

            num_vals_bins.append(len(vals))  # list of number of I04 values that were in each I05 increment (for errors)

        zero_args = np.where(y_i04 == 0)
        x_i05 = np.delete(x_i05, zero_args)
        y_i04 = np.delete(y_i04, zero_args)
        point_errs = np.delete(point_errs, zero_args)

        params, cov = sp.curve_fit(cubic, x_i05, y_i04)
        # perr = np.sqrt(np.diag(cov))

        xvalues = np.arange(min(x_i05), max(x_i05), 1)
        yvalues = cubic(xvalues, params[0], params[1], params[2], params[3])
        roots = np.roots([3 * params[0], 2 * params[1], params[2]])
        if np.iscomplex(roots).all():
            return
        gt = roots[~np.iscomplex(roots)][0].real

        # satellite_data_error = ??
        curve_fit_err = 0.5 * abs(3 * params[0] * gt ** 2 + 2 * params[1] * gt + params[2])
        gt_err = np.sqrt(curve_fit_err ** 2 + sum(x**2 for x in point_errs))


        if plot:
            plt.scatter(self.i04_flat, self.i05_flat, s=0.25)
            plt.plot(yvalues, xvalues, label="Line of best fit")

            #plt.errorbar(x_i05, y_i04, yerr=point_errs)
            plt.legend()
            plt.show()

        return gt, gt_err,params

    def draw(self, band="I04"):
        if band == "I04":
            plt.imshow(self.__i04)
            plt.show()
        else:
            plt.imshow(self.__i05)
            plt.show()
