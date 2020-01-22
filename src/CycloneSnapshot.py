import pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as npma
from matplotlib.widgets import RectangleSelector

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

    def __init__(self, I04: np.ndarray, I05: np.ndarray, pixel_x: int, pixel_y: int, sat_pos: float, metadata: dict,
                 M09: np.ndarray = None):
        self.__I04 = I04
        self.__I05 = I05
        self.M09 = M09
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
    def I04(self):
        if hasattr(self, "I04_mask"):
            return self.I04_mask
        else:
            return self.__I04

    @property
    def I05(self):
        if hasattr(self, "I05_mask"):
            return self.I05_mask
        else:
            return self.__I05

    @property
    def is_complete(self):
        """
        Check for any incomplete data in the Snapshot
        :return: Boolean, whether data contains NaN
        """
        return np.isnan(self.I04).any() or np.isnan(self.I05).any()

    @property
    def I05_celcius(self):
        return self.I05 - ABSOLUTE_ZERO

    def add_sub_snap(self, left, right, top, bottom, discrete=True):
        if discrete:
            I04_tmp = self.I04[left:right, bottom:top]
            I05_tmp = self.I05[left:right, bottom:top]
            self.sub_snaps[(left, right, top, bottom)] = CycloneSnapshot(I04_tmp, I05_tmp, self.pixel_x, self.pixel_y,
                                                                         self.satellite_azimuth, self.meta_data)

        return self.sub_snaps[(left, right, top, bottom)]

    def __discrete_img(self, fig, ax, band="I04"):
        da = self.I04 if band == "I04" else self.I05
        ax.imshow(da, origin="upper")

    def img_plot(self, fig, ax, band="I05"):
        if band == "I04":
            da = self.I04
        elif band == "I05":
            da = self.I05
        elif band == "M09":
            da = self.M09
        else:
            raise ValueError(f"Band: {band} is not available")
        im = ax.imshow(da, origin="upper",
                       extent=[-self.pixel_x * 0.5 * self.shape[1] / 1000,
                               self.pixel_x * 0.5 * self.shape[1] / 1000,
                               -self.pixel_y * 0.5 * self.shape[0] / 1000,
                               self.pixel_y * 0.5 * self.shape[0] / 1000])

        cb = plt.colorbar(im)
        if band == "M09":
            cb.set_label("Reflectance (%)")
        else:
            cb.set_label("Kelvin (K)")

    def scatter_plot(self, fig, ax, gt_fitter, fit=True):
        x = np.linspace(min(gt_fitter.i05), max(gt_fitter.i05))
        ax.scatter(gt_fitter.i04, gt_fitter.i05, s=0.1)
        if fit:
            gt, gt_err, params = gt_fitter.curve_fit_funcs()
            ax.plot([cubic(x_i, *params) for x_i in x], x, 'g-', label="Curve fit")
            ax.axhline(gt, xmin=min(x), xmax=max(x))
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_ylabel("Cloud Top Temperature (C)")
        ax.set_xlabel("I4 band reflectance (K)")

    def mask_thin_cirrus(self, reflectance_cutoff=50):
        """
        Use M9 band (if present) as a mask for the I04,I05 band above a given threshold.
        Ice present in high thin cirrus clouds will reflect light at a high altitude. In longer range bands I04,I05 this will make the pixel appear colder than the cloud top actually is
        :param reflectance_cutoff: Mask all pixels with a thin cirrus reflectance above this
        :return: None
        """
        if self.M09 is None:
            raise ValueError("No M9 data present")
        if hasattr(self, "I04_mask") or hasattr(self, "I05_mask"):
            self.I04_mask = npma.mask_or(self.I04_mask, npma.array(self.I04, mask=self.M09 >= reflectance_cutoff))
            self.I05_mask = npma.mask_or(self.I05_mask, npma.array(self.I05, mask=self.M09 >= reflectance_cutoff))
        self.I04_mask = npma.array(self.I04, mask=self.M09 >= reflectance_cutoff)
        self.I05_mask = npma.array(self.I05, mask=self.M09 >= reflectance_cutoff)

    def __flat(self, a):
        if isinstance(a, npma.MaskedArray):
            return a.compressed()
        elif isinstance(a, np.ndarray):
            return a.flatten()
        raise TypeError("a is not one of Masked Array or ndarray")

    def point_display(self):
        fig, ax = plt.subplots(1, 2)
        gt_fitter = GTFit(self.__flat(self.I04), self.__flat(self.I05))
        self.scatter_plot(fig, ax[0], gt_fitter, fit=False)
        self.__discrete_img(fig, ax[1])

        def __select_callback(eclick, erelease):
            ax[1].clear()
            self.__discrete_img(fig, ax[1])
            i4min, i4max = min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata)
            i5min, i5max = min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata)
            selected_points = np.argwhere(np.logical_and(np.logical_and(i4min < self.I04, self.I04 < i4max),
                                                         np.logical_and(i5min < self.I05, self.I05 < i5max)))
            ax[1].scatter([p[1] for p in selected_points], [p[0] for p in selected_points], s=0.15)

        def draw_cb(event):
            if draw_cb.RS.active:
                draw_cb.RS.update()

        draw_cb.RS = RectangleSelector(
            ax[0], __select_callback, drawtype="box", useblit=False,
            button=[1, 3], interactive=True, minspanx=5, minspany=5, spancoords="pixels"
        )
        draw_cb.RS.set_active(True)
        plt.connect("draw_event", draw_cb)
        plt.show()

    def plot(self, band="I04"):
        fig, ax = plt.subplots()
        self.img_plot(fig, ax, band)
        plt.show()

    def mask_array_I04(self, HIGH=273, LOW=210):
        if hasattr(self, "I04_mask") or hasattr(self, "I05_mask"):
            self.I04_mask = npma.mask_or(self.I04_mask, npma.masked_outside(self.__I04, LOW, HIGH))
        else:
            self.I04_mask = npma.masked_outside(self.__I04, LOW, HIGH)
        self.I05_mask = npma.array(self.__I05, mask=self.I04_mask.mask)

    def mask_array_I05(self, HIGH=273, LOW=210):
        if hasattr(self, "I04_mask") or hasattr(self, "I05_mask"):
            self.I05_mask = npma.mask_or(self.I05_mask, npma.masked_outside(self.__I05, LOW, HIGH))
        else:
            self.I05_mask = npma.masked_outside(self.__I05, LOW, HIGH)
        self.I04_mask = npma.array(self.__I04, mask=self.I04_mask.mask)

    def gt_fit(self):
        gt_fitter = GTFit(self.__flat(self.I04), self.__flat(self.I05))

        fig, ax = plt.subplots(1, 2)
        self.img_plot(fig, ax[1])
        self.scatter_plot(fig, ax[0], gt_fitter)
        plt.show()

    def unmask_array(self):
        del self.I04_mask
        del self.I05_mask

    def save(self, fpath):
        with open(fpath, "wb") as file:
            pickle.dump(self, file)


    def show_fitted_pixels(self):
        i05_flat = self.I05.flatten()
        i04_flat = self.I04.flatten()
        x_i05 = np.arange(220, 273, 1)
        if len(x_i05) < 1:
            return
        y_i04 = np.array([0] * len(x_i05))
        fig, axs = plt.subplots(1, 2)
        im = axs[1].imshow(self.I05)
        plt.colorbar(im)
        axs[1].set_title("%s %s" % (self.name, self.core_scene.start_time.strftime('%Y-%m-%d')))
        axs[0].scatter(i04_flat, i05_flat, s=0.25)
        for i, x in enumerate(x_i05):
            vals = i04_flat[np.where(np.logical_and(i05_flat > (x - 0.5), i05_flat < (x + 0.5)))]
            vals_5_min = []
            vals_5_min_i05val = []
            if len(vals) < 1:
                continue
            percent_range = int(np.ceil(len(vals) * 0.075))
            if x > 235:  # Takes minimum values lower than theoretical min gt and v.v.
                for j in range(percent_range):
                    if len(vals) == 0:  # Not all values of i05 will have 5 i04 values
                        break

                    vals_5_min.append(min(vals))
                    i05s_with_same_i04 = i05_flat[np.where(i04_flat == min(vals))]
                    for i05 in i05s_with_same_i04:
                        if x - 0.5 < i05 < x + 0.5:
                            vals_5_min_i05val.append(i05)
                            if len(vals_5_min_i05val) > j:
                                break

                    vals = np.delete(vals, np.where(vals == min(vals)))
                y_i04[i] = np.median(vals_5_min)
                axs[0].scatter(vals_5_min, vals_5_min_i05val, color="orange", s=5)
                # rect = self.rects[key]
                # offset_x = (self.width - self.rects[key].width/self.pixel_x)/2
                # offset_y = (self.height - self.rects[key].height/self.pixel_y)/2
                for xy in range(len(vals_5_min)):
                    points = np.argwhere(np.logical_and(self.I05 == vals_5_min_i05val[xy], self.I04 == vals_5_min[xy]))
                    axs[1].scatter([p[1] for p in points], [p[0] for p in points], s=5, c="red")
            else:
                increasing_range = int(np.ceil(len(vals) * (0.075 + (235 - x) * 0.025)))
                for j in range(percent_range):
                    if len(vals) == 0:  # Not all values of i05 will have 5 i04 values
                        break
                    vals.sort()
                    idx = int((235 - x) * 0.025 * len(vals) + j)  # changes idx to shift 2.5% range every x value
                    vals_5_min.append(vals[idx])
                    i05s_with_same_i04 = i05_flat[np.where(i04_flat == vals[idx])]
                    for i05 in i05s_with_same_i04:
                        if x - 0.5 < i05 < x + 0.5:
                            vals_5_min_i05val.append(i05)
                            if len(vals_5_min_i05val) > j:
                                break

                y_i04[i] = np.median(vals_5_min)
                axs[0].scatter(vals_5_min, vals_5_min_i05val, color="black", s=5)
                # rect = self.rects[key]
                # offset_x = (self.width - self.rects[key].width / self.pixel_x) / 2
                # offset_y = (self.height - self.rects[key].height / self.pixel_y) / 2
                for xy in range(len(vals_5_min)):
                    points = np.argwhere(np.logical_and(self.I05 == vals_5_min_i05val[xy], self.I04 == vals_5_min[xy]))
                    axs[1].scatter([p[1] for p in points], [p[0] for p in points], s=5, c="black")

        zero_args = np.where(y_i04 == 0)
        x_i05 = np.delete(x_i05, zero_args)
        y_i04 = np.delete(y_i04, zero_args)

        params, cov = sp.curve_fit(cubic, x_i05, y_i04, absolute_sigma=True)

        xvalues = np.arange(min(x_i05), max(x_i05), 1)
        yvalues = cubic(xvalues, *params)

        gt_ve = (-params[1] + np.sqrt(params[1] ** 2 - 3 * params[0] * params[2])) / (3 * params[0])
        if np.iscomplex(gt_ve) or min(x_i05) > gt_ve > max(x_i05):
            return
        self.gt = [gt_ve]

        axs[0].plot(yvalues, xvalues, color="r")
        axs[0].invert_xaxis()
        axs[0].invert_yaxis()
        if 300 > gt_ve > 200:
            axs[0].axhline(gt_ve, color="r")
        plt.show()
        return gt_ve

    def half_eye(self):
        I05_flat = self.I05.flatten()
        I04_flat = self.I04.flatten()
        rows = len(self.I05)
        columns = len(self.I05[0])
        half_mask = np.zeros(np.shape(self.I05))
        for i in range(int(rows / 2)):
            for j in range(int(columns)):
                half_mask[i][j] = 1
        masked_I05 = np.ma.masked_outside(self.I05, 220, 270)
        # masked_I05 = np.ma.masked_array(masked_I05, mask=half_mask)
        masked_I04 = self.I04
        masked_rows = len(masked_I05)
        masked_columns = len(masked_I05[0])
        print(masked_rows, masked_columns)
        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(self.I04, self.I05, s=0.25)
        axs[0].scatter(masked_I04, masked_I05, s=0.5, c="black")
        axs[1].imshow(self.I05)
        axs[1].imshow(masked_I05, cmap="Dark2")
        axs[0].invert_xaxis()
        axs[0].invert_yaxis()
        plt.show()
