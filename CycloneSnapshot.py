import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as npma
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from scipy.stats import sem

from GTFit import GTFit

ABSOLUTE_ZERO = 273.15
NM_TO_M = 1852
R_E = 6371000


def wrap_360(x):
    if x < 0:
        return x + 360
    elif x > 360:
        return x - 360
    else:
        return x


class CycloneSnapshot:
    """
    Uniformly gridded single snapshot of cyclone.
    """

    __slots__ = ["__I04", "__I05", "I04_mask", "I05_mask", "I01", "M09", "pixel_x", "pixel_y", "b_lon", "b_lat",
                 "metadata", "solar_zenith", "satellite_azimuth"]

    @staticmethod
    def load(fpath):
        with open(fpath, "rb") as file:
            cs = pickle.load(file)
        return cs

    def __init__(self, I04: np.ndarray, I05: np.ndarray, pixel_x: int, pixel_y: int, sat_pos: np.ndarray,
                 metadata: dict,
                 b_lon, b_lat,
                 M09: np.ndarray = None, I01: np.ndarray = None,
                 solar: np.ndarray = None):
        self.__I04 = I04
        self.__I05 = I05
        if np.isnan(I04).any():
            self.I04_mask = npma.masked_invalid(I04)
        if np.isnan(I05).any():
            self.I05_mask = npma.masked_invalid(I05)
        self.M09 = M09
        self.I01 = I01
        assert self.I04.shape == self.I05.shape

        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.b_lon = b_lon
        self.b_lat = b_lat

        self.metadata = dict(metadata)
        self.solar_zenith = solar
        self.satellite_azimuth = sat_pos


    @property
    def shape(self):
        return self.I04.shape

    @property
    def width(self):
        return self.pixel_x * self.shape[1] / (NM_TO_M * 60)

    @property
    def height(self):
        return self.pixel_y * self.shape[0] / (NM_TO_M * 60)

    @property
    def is_shaded(self):
        return self.image_mean_azimuth < 180

    @property
    def image_mean_azimuth(self):
        from CycloneImage import wrap
        return wrap(self.satellite_azimuth.mean())

    @property
    def I04(self):
        if hasattr(self, "I04_mask"):
            return self.I04_mask
        else:
            return self.__I04

    @property
    def t_lat(self):
        return self.b_lat + self.height

    def distance_to_snap(self, snap: "CycloneSnapshot", units="m"):
        angle = np.arccos(
            np.sin(np.deg2rad(self.b_lat)) * np.sin(np.deg2rad(snap.b_lat)) + np.cos(np.deg2rad(self.b_lat)) * np.cos(
                np.deg2rad(snap.b_lat)) * np.cos(
                np.abs(np.deg2rad(snap.b_lon - self.b_lon))))
        if units == "m":
            return angle * R_E
        if units == "nm":
            return angle * R_E / NM_TO_M

    @property
    def t_lon(self):
        return self.b_lon + self.width

    @property
    def I05(self):
        if hasattr(self, "I05_mask"):
            return self.I05_mask
        else:
            return self.__I05

    def check_overlap(self, other: "CycloneSnapshot"):
        """
        Check whether other CycloneSnapshot object is overlap
        :param other:
        :return:
        """
        if self.b_lon > other.t_lon or other.b_lon > self.t_lon:
            return False
        if self.b_lat > other.t_lat or other.b_lat > self.t_lat:
            return False
        return True

    @property
    def is_complete(self):
        """
        Check for any incomplete data in the Snapshot
        :return: Boolean, whether data contains NaN
        """
        return np.isnan(self.I04).any() or np.isnan(self.I05).any()

    def celcius(self, a):
        return a - ABSOLUTE_ZERO

    def add_sub_snap_edge(self, left, right, bottom, top, b_lon, b_lat):
        I04_tmp = self.I04[bottom:top, left:right]
        I05_tmp = self.I05[bottom:top, left:right]
        cs = CycloneSnapshot(I04_tmp, I05_tmp, self.pixel_x, self.pixel_y,
                             self.satellite_azimuth[bottom:top, left:right], self.metadata,
                             b_lon,
                             b_lat,
                             M09=self.M09[bottom:top, left:right] if self.M09 is not None else None,
                             I01=self.I01[bottom:top, left:right],
                             solar=self.solar_zenith[bottom:top, left:right] if self.solar_zenith is not None else None)

        return cs

    def add_sub_snap_origin(self, x_i, y_i, width, height, lon, lat):
        return self.add_sub_snap_edge(int(x_i - width / 2), int(x_i + width / 2), int(y_i - height / 2),
                                      int(y_i + height / 2), lon, lat)

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
        elif band == "I01":
            da = self.I01
        else:
            raise ValueError(f"Band: {band} is not available")
        im = ax.imshow(da, origin="upper",
                       extent=[-self.pixel_x * 0.5 * self.shape[1] / 1000,
                               self.pixel_x * 0.5 * self.shape[1] / 1000,
                               -self.pixel_y * 0.5 * self.shape[0] / 1000,
                               self.pixel_y * 0.5 * self.shape[0] / 1000])
        ax.set_title("%s %s" % (self.metadata["NAME"], self.metadata["ISO_TIME"]))
        ax.set_xlabel("km")
        ax.set_ylabel("km")
        cb = plt.colorbar(im)

        if band == "M09" or band == "I01":
            cb.set_label("Reflectance (%)")
        else:
            cb.set_label("Kelvin (K)")

    def scatter(self, fig, ax):
        ax.scatter(self.flat(self.I04), self.celcius(self.flat(self.I05)), s=0.1)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_ylabel("Cloud Top Temperature (C)")
        ax.set_xlabel("I4 band reflectance (K)")

    @property
    def quadrant(self):
        eye_azimuth = wrap_360(np.rad2deg(np.arctan2(self.b_lon + (self.width / 2) - self.metadata["USA_LON"],
                                                     self.b_lat + self.height / 2 - self.metadata["USA_LAT"])))
        if 0 <= wrap_360(eye_azimuth - self.metadata["STORM_DIR"]) < 90:
            return "RF"
        elif 90 <= wrap_360(eye_azimuth - self.metadata["STORM_DIR"]) < 180:
            return "RB"
        elif 180 <= wrap_360(eye_azimuth - self.metadata["STORM_DIR"]) < 270:
            return "LB"
        elif 270 <= wrap_360(eye_azimuth - self.metadata["STORM_DIR"]) < 360:
            return "LF"
        else:
            raise ValueError(
                f"Value for azimuthal offset does not make sense. Expecting a value between 0 and 360 degrees, received {wrap_360(eye_azimuth - self.metadata['STORM_DIR'])}")

    def mask_using_I01(self, reflectance_cutoff=80):
        """
        Use I01 band (if present) to mask dimmer pixels below a certain reflectance
        """
        if self.I01 is None:
            raise ValueError("No I1 data present")
        if hasattr(self, "I04_mask") or hasattr(self, "I05_mask"):
            new_I04_mask = npma.mask_or(self.I04_mask.mask,
                                        npma.array(self.I04, mask=self.I01 <= reflectance_cutoff).mask)
            new_I05_mask = npma.mask_or(self.I05_mask.mask,
                                        npma.array(self.I05, mask=self.I01 <= reflectance_cutoff).mask)
            self.I04_mask = npma.array(self.__I04, mask=new_I04_mask)
            self.I05_mask = npma.array(self.__I05, mask=new_I05_mask)
        else:
            self.I04_mask = npma.array(self.I04, mask=self.I01 <= reflectance_cutoff)
            self.I05_mask = npma.array(self.I05, mask=self.I01 <= reflectance_cutoff)

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
            new_I04_mask = npma.mask_or(self.I04_mask.mask,
                                        npma.array(self.I04, mask=self.M09 >= reflectance_cutoff).mask)
            new_I05_mask = npma.mask_or(self.I05_mask.mask,
                                        npma.array(self.I05, mask=self.M09 >= reflectance_cutoff).mask)
            self.I04_mask = npma.array(self.__I04, mask=new_I04_mask)
            self.I05_mask = npma.array(self.__I05, mask=new_I05_mask)
        else:
            self.I04_mask = npma.array(self.I04, mask=self.M09 >= reflectance_cutoff)
            self.I05_mask = npma.array(self.I05, mask=self.M09 >= reflectance_cutoff)

    def flat(self, a):
        if isinstance(a, npma.MaskedArray):
            return a.compressed()
        elif isinstance(a, np.ndarray):
            return a.flatten()
        raise TypeError("a is not one of Masked Array or ndarray")

    def point_display(self):
        fig, ax = plt.subplots(1, 2)

        self.scatter(fig, ax[0])
        self.__discrete_img(fig, ax[1])

        def __select_callback(eclick, erelease):
            ax[1].clear()
            self.__discrete_img(fig, ax[1])
            i4min, i4max = min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata)
            i5min, i5max = min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata)
            i4_points = np.logical_and(i4min < self.I04, self.I04 < i4max)
            i5_points = np.logical_and(i5min < self.celcius(self.I05), self.celcius(self.I05) < i5max)
            selected_points = np.argwhere(np.logical_and(i4_points, i5_points))
            ax[1].scatter([p[1] for p in selected_points], [p[0] for p in selected_points], s=0.5, c="r")
            fig.canvas.draw()

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

    def plot(self, band="I05", show=True, save_dir=None):
        fig, ax = plt.subplots()
        self.img_plot(fig, ax, band)
        if save_dir:
            plt.savefig(save_dir)
            plt.close()
        if show:
            plt.show()

    def mask_array_I04(self, HIGH=273, LOW=230):
        if hasattr(self, "I04_mask") or hasattr(self, "I05_mask"):
            new_I04_mask = npma.mask_or(self.I04_mask.mask, npma.masked_outside(self.__I04, LOW, HIGH).mask)
            new_I05_mask = npma.mask_or(self.I05_mask.mask, npma.masked_outside(self.__I04, LOW, HIGH).mask)
            self.I05_mask = npma.array(self.__I05, mask=new_I05_mask)
            self.I04_mask = npma.array(self.__I04, mask=new_I04_mask)
        else:
            self.I04_mask = npma.masked_outside(self.__I04, LOW, HIGH)
            self.I05_mask = npma.array(self.__I05, mask=self.I04_mask.mask)

    def mask_array_I05(self, HIGH=273, LOW=230):
        if hasattr(self, "I04_mask") or hasattr(self, "I05_mask"):
            new_I05_mask = npma.mask_or(self.I05_mask.mask, npma.masked_outside(self.__I05, LOW, HIGH).mask)
            new_I04_mask = npma.mask_or(self.I04_mask.mask, npma.masked_outside(self.__I05, LOW, HIGH).mask)
            self.I05_mask = npma.array(self.__I05, mask=new_I05_mask)
            self.I04_mask = npma.array(self.__I04, mask=new_I04_mask)
        else:
            self.I05_mask = npma.masked_outside(self.__I05, LOW, HIGH)
            self.I04_mask = npma.array(self.__I04, mask=self.I05_mask.mask)

    def mask_half(self, half="right"):
        blank_mask = np.zeros_like(self.__I05)
        if half == "top":
            idx = np.arange(0, int(len(self.__I05) / 2))
            blank_mask[idx, :] = 1
        if half == "left":
            idx = np.arange(0, int(len(self.__I05) / 2))
            blank_mask[:, idx] = 1
        if half == "bottom":
            idx = np.arange(int(len(self.__I05) / 2), len(self.__I05))
            blank_mask[idx, :] = 1
        if half == "right":
            idx = np.arange(int(len(self.__I05) / 2), len(self.__I05))
            blank_mask[:, idx] = 1
        blank_mask = blank_mask.astype("bool")
        if hasattr(self, "I04_mask") or hasattr(self, "I05_mask"):
            new_I05_mask = npma.mask_or(self.I05_mask.mask, blank_mask)
            new_I04_mask = npma.mask_or(self.I04_mask.mask, blank_mask)
            self.I05_mask = npma.array(self.__I05, mask=new_I05_mask)
            self.I04_mask = npma.array(self.__I04, mask=new_I04_mask)
        else:
            self.I05_mask = npma.array(self.__I05, mask=blank_mask)
            self.I04_mask = npma.array(self.__I04, mask=blank_mask)

    def gt_piece_all(self, plot=True, raise_up=0, raise_lower=-40, save_fig=None, show=True, overlap=None):
        if overlap:
            if self.check_overlap(overlap):
                return np.nan, np.nan, np.nan
        if isinstance(self.I04, npma.MaskedArray):
            i01 = np.ma.array(self.I01, mask=self.I04.mask)
        else:
            i01 = self.I01

        gt_fitter = GTFit(self.flat(self.I04), self.celcius(self.flat(self.I05)), self.flat(i01))
        try:
            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(9, 6))
                self.img_plot(fig, ax[1])
                (gt, gt_err), (r2, params) = gt_fitter.piecewise_fit(fig, ax[0])
                if save_fig:
                    plt.savefig(save_fig)
                    plt.close("all")
                if show:
                    plt.show()
            else:
                (gt, gt_err), (r2, params) = gt_fitter.piecewise_fit()
            if raise_up < gt or gt + gt_err < raise_lower or r2 < 0.85:
                raise ValueError(f"Glaciation Temperature Outside of range: {gt}")
            return gt, gt_err, r2
        except (RuntimeError, ValueError):
            return np.nan, np.nan, np.nan

    def gt_piece_percentile(self, percentile=5, plot=True, raise_up=0, raise_lower=-38, save_fig=None, show=True,
                            overlap=None):
        gt_fitter = GTFit(self.flat(self.I04), self.celcius(self.flat(self.I05)))
        if overlap:
            if self.check_overlap(overlap):
                return np.nan, np.nan, np.nan
        try:
            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(9, 6))
                self.img_plot(fig, ax[1])
                (gt, gt_err), (r2, params) = gt_fitter.piecewise_percentile(percentile=percentile, fig=fig, ax=ax[0])
                if save_fig:
                    plt.savefig(save_fig)
                    plt.close("all")
                if show:
                    plt.show()
            else:
                (gt, gt_err), (r2, params) = gt_fitter.piecewise_percentile(percentile=percentile)
            if raise_up + gt_err * 2 < gt or gt - gt_err * 2 < raise_lower or r2 < 0.85:  # Sanity check
                raise ValueError("Outside predefined range")
            return gt, gt_err, r2
        except (RuntimeError, ValueError):
            return np.nan, np.nan, np.nan

    def unmask_array(self):
        if np.isnan(self.__I04).any():
            self.I04_mask = npma.masked_invalid(self.__I04)
        else:
            del self.I04_mask
        if np.isnan(self.__I05).any():
            self.I05_mask = npma.masked_invalid(self.__I05)
        else:
            del self.I05_mask

    def save(self, fpath):
        with open(fpath, "wb") as file:
            pickle.dump(self, file)

    def plot_solar(self):
        solar_im = plt.imshow(self.solar_zenith)
        plt.colorbar(solar_im)

    def plot_sat(self):
        sat_im = plt.imshow(self.satellite_azimuth)
        plt.colorbar(sat_im)

    def mask_diff_sat_sun_zenith(self, threshold=61):
        data = np.abs(self.solar_zenith - self.satellite_azimuth)
        if hasattr(self, "I04_mask") or hasattr(self, "I05_mask"):
            new_I04_mask = npma.mask_or(self.I04_mask.mask,
                                        npma.array(self.I04, mask=data >= threshold).mask)
            new_I05_mask = npma.mask_or(self.I05_mask.mask,
                                        npma.array(self.I05, mask=data >= threshold).mask)
            self.I04_mask = npma.array(self.__I04, mask=new_I04_mask)
            self.I05_mask = npma.array(self.__I05, mask=new_I05_mask)
        else:
            self.I04_mask = npma.array(self.I04, mask=data >= threshold)
            self.I05_mask = npma.array(self.I05, mask=data >= threshold)


class SnapshotGrid:
    def __init__(self, gd: List[List[CycloneSnapshot]], imageInstance=None):
        self.grid = gd
        self.height = len(gd)
        self.width = len(gd[0])
        self.imageInstance = imageInstance
        if imageInstance:
            self.vals = {}
            for k, v in self.imageInstance.metadata.items():
                if k == "ISO_TIME":
                    self.vals[k] = v.strftime("%Y-%m-%d %H:%M:%S")
                    continue
                self.vals[k] = v

    @property
    def corners(self) -> List[CycloneSnapshot]:
        return [self.grid[0][0], self.grid[0][-1], self.grid[-1][0], self.grid[-1][-1]]

    def set_eye_gt(self, gt, gt_err):
        self.vals["EYE"] = gt
        self.vals["EYE_ERR"] = gt_err

    def plot_all(self, band):
        fig, axs = plt.subplots(self.width, self.height)
        for i, row in enumerate(self.grid):
            for j, snap in enumerate(row):
                snap.img_plot(fig, axs[i][j], band)
        plt.show()

    def mask_all_I05(self, LOW=220, HIGH=290):
        for row in self.grid:
            for snap in row:
                snap.mask_array_I04(LOW=LOW, HIGH=HIGH)

    def save(self):
        import json
        with open(os.path.join(self.imageInstance.get_dir(), "out.json"), "w") as f:
            json.dump(self.vals, f)

    def piecewise_glaciation_temperature(self, plot=True, show=True, save=False):
        gd = [[list(snap.gt_piece_percentile(plot=False, overlap=self.imageInstance.eye)) for snap in row] for row in
              self.grid]

        self.gt_grid, self.gt_err, self.r2 = np.squeeze(np.split(np.array(gd), 3, 2))
        if plot:
            fig, ax = plt.subplots()
            im = ax.imshow(self.gt_grid, origin="upper")
            cb = plt.colorbar(im)
            cb.set_label("Glaciation Temperature (C)")
            if save:
                plt.savefig(os.path.join(self.imageInstance.get_dir(), "piecewise_gt_distribution.png"))
                plt.close()
            if show:
                plt.show()
            fig, ax = plt.subplots()
            im2 = ax.imshow(self.r2, origin="upper")
            cb2 = plt.colorbar(im2)
            cb2.set_label("R^2 goodness of fit coefficient")
            if save:
                plt.savefig(os.path.join(self.imageInstance.get_dir(), "piecewise_r2_distribution.png"))
                plt.close()
            if show:
                plt.show()
        self.vals["GT_GRID"] = self.gt_grid[~np.isnan(self.gt_grid)].tolist()
        self.vals["GT_MEAN"] = np.nanmean(self.gt_grid)
        self.vals["GT_STD"] = np.nanstd(self.gt_grid)

    def get_mean_r2(self):
        try:
            return np.nanmean(self.r2)
        except AttributeError:
            self.piecewise_glaciation_temperature(plot=False)
            self.get_mean_r2()

    def get_mean_gt(self):
        try:
            return np.nanmean(self.gt_grid)
        except AttributeError:
            self.piecewise_glaciation_temperature(plot=False)
            self.get_mean_gt()

    def radial_distribution(self, show=True, save=False):
        fig, ax = plt.subplots()
        ax.scatter(
            [snap.distance_to_snap(self.imageInstance.eye) / 1000 for row in self.grid for snap in row if snap != 0],
            self.gt_grid.flatten())
        ax.invert_yaxis()
        ax.set_ylabel("Glaciation Temperature (C)")
        ax.set_xlabel("Radial distance from eye (km)")
        if save:
            plt.savefig(os.path.join(self.imageInstance.get_dir(), "radial_distribution.png"))
            plt.close()
        if show:
            plt.show()

    # def gt_radial_step_distr(self, radial_step, eye_gt=0, eye_gt_err=0, plot=True, save=False, show=True):
    #     # Check each corner for maximum radial distance to the eye
    #     max_r = np.inf
    #     for c in self.corners:
    #         dist = c.distance_to_snap(self.imageInstance.eye)
    #         if dist < max_r:
    #             max_r = dist
    #
    #     radial_steps = np.arange(0, max_r, radial_step)
    #     bins = [[] for i in range(len(radial_steps) - 1)]
    #     for i, row in enumerate(self.grid):
    #         for j, snap in enumerate(row):
    #             if np.isnan(self.gt_grid[i][j]):
    #                 continue
    #             distance = snap.distance_to_snap(self.imageInstance.eye)
    #             bins[distance // radial_step].append(self.gt_grid[i][j])
    #
    #     for i, r in enumerate(radial_steps[1:]):
    #         self.vals[f"{radial_steps[i - 1]}-{r}"] = np.nanmean(bins[i - 1])
    #         self.vals[f"{radial_steps[i - 1]}-{r}_ERR"] = sem(bins[i - 1])
    #
    #     if plot:
    #         fig, ax = plt.subplots()
    #         rects = ax.errorbar(radial_steps,
    #                             [self.vals[f"{radial_steps[i - 1]}-{radial_steps[i]}"] for i in
    #                              range(1, len(radial_steps))].insert(0, self.vals["EYE"]),
    #                             capsize=5,
    #                             yerr=[self.vals[f"{radial_steps[i - 1]}-{radial_steps[i]}_ERR"] for i in
    #                                   range(1, len(radial_steps))].insert(0, self.vals["EYE_ERR"]))
    #         # ax.set_xticklabels() TODO: Get the labels working
    #         ax.set_ylabel("Glaciation Temperature (C)")
    #         ax.invert_yaxis()
    #         ax.set_ylim(bottom=0, top=-45)
    #         ax.set_title(
    #             f"{self.imageInstance.metadata['NAME']} on {self.imageInstance.metadata['ISO_TIME']}\nGlaciation Temperature over radius")

    @property
    def valid_cells(self):
        return [ci for row in self.grid for ci in row if not np.isnan(ci.gt_piece_percentile(plot=False)[0])]

    def histogram_from_eye(self, show=True, save=False):
        gt_from_eye = self.gt_grid.flatten() - self.vals["EYE"]
        gt_from_eye = gt_from_eye[~np.isnan(gt_from_eye)]
        fig, ax = plt.subplots()
        ax.hist(gt_from_eye, bins=10)
        ax.set_xlabel("T_g of external cell - T_g of the eye ")
        if save:
            plt.savefig(os.path.join(self.imageInstance.get_dir(), "eye_t_offset_hist.png"))
            plt.close(fig)
        if show:
            plt.show()

    def gt_quadrant_distribution(self, plot=True, save=False, show=True):
        """
        Plot distribution of the glaciation temperature in the four quadrants of the cyclone.
        If eye_gt is passed then will compare this against the glaciation temperature of the eye for visualisation
        :param ey_gt: Glaciation temperature of the eye in celsius
        :return: None
        """
        distr = {"LF": [], "RF": [], "RB": [], "LB": []}
        for i, row in enumerate(self.grid):
            for j, snap in enumerate(row):
                if np.isnan(self.gt_grid[i][j]) or self.gt_grid[i][j] == 0:
                    continue
                distr[snap.quadrant].append(self.gt_grid[i][j])
        for k, v in distr.items():
            self.vals[k] = np.nanmean(v)

        vals_err = {k: sem(np.array(v)) for k, v in distr.items()}
        vals_err["EYE"] = self.vals["EYE_ERR"]

        if plot:
            fig, ax = plt.subplots()

            rects = ax.bar(range(5),
                           [self.vals["LB"], self.vals["RB"], self.vals["RF"], self.vals["LF"], self.vals["EYE"]],
                           align="center", yerr=vals_err.values(), capsize=5)
            ax.set_xticks(range(5))
            ax.set_xticklabels(["LB", "RB", "RF", "LF", "EYE"])
            ax.set_ylabel("Glaciation Temperature (C)")
            ax.invert_yaxis()
            ax.set_ylim(bottom=0, top=-45)

            ax.set_title(
                f"{self.imageInstance.metadata['NAME']} on {self.imageInstance.metadata['ISO_TIME']}\nGlaciation Temperature Distribution by Quadrant")

            for rect in rects:
                ax.annotate(f"{round(rect.get_height(), 2)}",
                            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom")
            if save:
                plt.savefig(os.path.join(self.imageInstance.get_dir(), "quadrant_plot.png"))
                plt.close()
            q_fig, q_ax = plt.subplots(2, 2, figsize=(8, 6))
            for i, quad in enumerate(("LB", "RB", "LF", "LB")):
                if len(distr[quad]) < 1:
                    continue
                q_ax[i // 2][i % 2].hist(distr[quad])
                q_ax[i // 2][i % 2].set_ylabel("Frequency")
                q_ax[i // 2][i % 2].set_xlabel("Glaciation Temperature (C)")
            if save:
                plt.savefig(os.path.join(self.imageInstance.get_dir(), "quadrant_distribution.png"))
                plt.close()
            if show:
                plt.show()

        for k, v in vals_err.items():
            if k == "EYE":
                continue
            self.vals[f"{k}_ERR"] = v
            self.vals[f"{k}_COUNT"] = len(distr[k])

        print(
            f"Number of grid cells per quadrant\nLF:{self.vals['LF_COUNT']}\nRF:{self.vals['RF_COUNT']}\nRB:{self.vals['RB_COUNT']}\nLB:{self.vals['LB_COUNT']}")

        return self.vals
