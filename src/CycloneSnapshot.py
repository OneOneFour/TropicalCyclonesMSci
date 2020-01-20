import pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as npma

from GTFit import GTFit, cubic

ABSOLUTE_ZERO = 273.15


class CycloneSnapshot:
    """
    Uniformly gridded single snapshot of cyclone.
    """

    @staticmethod
    def load(fpath):
        with open(fpath) as file:
            cs = pickle.load(file)
        return cs

    def __init__(self, I04: np.ndarray, I05: np.ndarray, pixel_x: int, pixel_y: int, sat_pos: float, metadata: dict):
        self.I04 = I04
        self.I05 = I05
        assert self.I04.shape == self.I05.shape
        self.shape = self.I04.shape
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.meta_data = dict(metadata)
        self.satellite_azimuth = sat_pos
        self.sub_snaps = {}

    @property
    def is_eyewall_shaded(self):
        return self.satellite_azimuth > 180

    @property
    def is_complete(self):
        """
        Check for any incomplete data in the Snapshot
        :return: Boolean, whether data contains NaN
        """
        return np.isnan(self.I04).any() or np.isnan(self.I05).any()

    @property
    def I05_celcius(self):
        if hasattr(self, "I05_mask"):
            return self.I05_mask - ABSOLUTE_ZERO
        else:
            return self.I05 - ABSOLUTE_ZERO

    def add_sub_snap(self, left, right, top, bottom, discrete=True):
        if discrete:
            I04_tmp = self.I04[left:right, bottom:top]
            I05_tmp = self.I05[left:right, bottom:top]
            self.sub_snaps[(left, right, top, bottom)] = CycloneSnapshot(I04_tmp, I05_tmp, self.pixel_x, self.pixel_y)

        return self.sub_snaps[(left, right, top, bottom)]

    def img_plot(self, fig, ax, band="I04"):
        da = self.I04 if band == "I04" else self.I05
        im = ax.imshow(da, origin="upper",
                       extent=[-self.pixel_x * 0.5 * self.shape[0],
                               self.pixel_x * 0.5 * self.shape[0],
                               -self.pixel_y * 0.5 * self.shape[1],
                               self.pixel_y * 0.5 * self.shape[1]])
        cb = plt.colorbar(im)
        cb.set_label("Kelvin (K)")

    def scatter_plot(self, fig, ax, gt=None, gt_params=None):
        if hasattr(self, "I04_mask"):
            x = np.linspace(min(self.I05_celcius.compressed()), max(self.I05_celcius.compressed()))
            ax.scatter(self.I04_mask.compressed(), self.I05_celcius.compressed(), s=0.1)
        else:
            x = np.linspace(min(self.I05_celcius.flatten()), max(self.I05_celcius.flatten()))
            ax.scatter(self.I04.flatten(), self.I05_celcius.flatten(), s=0.1)
        # ax.plot([cubic(x_i, *gt_params) for x_i in x], 'g-', label="Curve fit")
        ax.axhline(gt, xmin=min(x), xmax=max(x))
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_ylabel("Cloud Top Temperature (K)")
        ax.set_xlabel("I4 band reflectance (K)")

    def plot(self, band="I04"):
        fig, ax = plt.subplots()
        self.img_plot(fig, ax, band)
        plt.show()

    def mask_array(self, HIGH=273, LOW=210):
        self.I05_mask = npma.masked_outside(self.I05, LOW, HIGH)
        self.I04_mask = npma.array(self.I04, mask=self.I05_mask.mask)

    def gt_fit(self):
        if hasattr(self, "I05_mask"):
            gt_fitter = GTFit(self.I04_mask.compressed(), self.I05_celcius.compressed())
        else:
            gt_fitter = GTFit(self.I04.flatten(), self.I05_celcius.flatten())

        gt, gt_err, coeffs = gt_fitter.curve_fit_modes("mean")

        fig, ax = plt.subplots(1, 2)
        self.img_plot(fig, ax[0])
        self.scatter_plot(fig, ax[1], gt, coeffs)
        plt.show()

    def save(self, fpath):
        with open(fpath, "wb") as file:
            pickle.dump(self, file)
