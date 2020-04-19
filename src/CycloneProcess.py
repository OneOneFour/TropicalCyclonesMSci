import glob
import os
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
from tqdm import tqdm

from BestTrack import best_track_df
from GTFit import GTFit
from alt.CycloneMap import CycloneImageFast

font = {'family': 'normal',
        'size': 15}

matplotlib.rc("font", **font)

CACHE_DIRECTORY = os.environ["CACHE_DIRECTORY"]

# font = {'family': 'normal',
# #         'weight': 'bold',
# #         'size': 15}
# # matplotlib.rc('font', **font)
plt.ioff()


def get_full_column(df: pd.DataFrame, key: str):
    if key in df:
        a = [val for cell in df[key].values.flatten() for val in cell]
        return a
    raise KeyError


def compare_histograms(df: pd.DataFrame):
    fig, axs = plt.subplots(1, 2)
    # Eye histogram
    eyewall_hist(fig,axs[0])

    external_hist(fig,axs[1])
    plt.show()


def eyewall_hist(df, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.hist(df["GT_EYEWALL"],bins=np.arange(273.15-44,273.15-10,2),rwidth=0.8,histtype="barstacked")
    ax.set_title("Eyewall $T_g$")
    ax.set_xlabel("Glaciation Temperature (K)")
    ax.set_ylabel("Frequency")
    plt.show()

def external_hist(df, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.hist(get_full_column(df, "EXTERNAL_GT"),bins=np.arange(-44,-10,2))
    ax.set_title("Environmental cell $T_g$")
    ax.set_xlabel("Glaciation Temperature (K)")
    ax.set_ylabel("Frequency")
    plt.show()

def bin_mean(df, key_i5, key_i4):
    i5 = list(chain.from_iterable(df[key_i5]))
    i4 = list(chain.from_iterable(df[key_i4]))
    gt_fitter = GTFit(i4, i5)
    return gt_fitter.bin_data(np.mean)


def windspeed_vs_TG_eyewall(df):
    fig, ax = plt.subplots()
    ax.scatter(df["GT_EYEWALL"], df["USA_WIND"])
    ax.set_xlabel("Glaciation Temperature (C)")
    ax.set_ylabel("Wind Speed (kts)")
    plt.show()


def compare_windspeed(df):
    fig, ax = plt.subplots()
    ax.hist(df["USA_WIND"])
    ax.set_xlabel("Wind Speed (Kts)")
    ax.set_ylabel("Frequency")
    plt.show()


def plot_binned_i5vsi4r(df, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df[f"{p}_EYEWALL_REF_I5"]))
        reflectance = list(chain.from_iterable(df[f"{p}_EYEWALL_REF_I4"]))
        gt_fit_i5_ref = GTFit(reflectance, i5)
        i5, reflectance = gt_fit_i5_ref.bin_data(np.mean)
        ax.plot(reflectance, i5, label=f"{p}th percentile eyewall")
        i5 = list(chain.from_iterable(df[f"{p}_EXTERNAL_REF_I5"]))
        reflectance = list(chain.from_iterable(df[f"{p}_EXTERNAL_REF_I4"]))
        gt_fit_i5_ref = GTFit(reflectance, i5)
        i5, reflectance = gt_fit_i5_ref.bin_data(np.mean)
        ax.plot(reflectance, i5, label=f"{p}th percentile external")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("I4 band reflectance")
    ax.set_ylabel("Cloud top temperature (K)")
    ax.legend()
    plt.show()


def plot_binned_i5vsi4r_both(df_4, df_5, percentiles, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df_4[f"{p}_EYEWALL_REF_I5"]))
        reflectance = list(chain.from_iterable(df_4[f"{p}_EYEWALL_REF_I4"]))
        gt_fit_i5_ref = GTFit(reflectance, i5)
        i5, reflectance = gt_fit_i5_ref.bin_data(np.mean)
        ax.plot(reflectance, i5, label=f"Cat 5 eyewall, {p}th percentile")
        i5 = list(chain.from_iterable(df_4[f"{p}_EXTERNAL_REF_I5"]))
        reflectance = list(chain.from_iterable(df_4[f"{p}_EXTERNAL_REF_I4"]))
        gt_fit_i5_ref = GTFit(reflectance, i5)
        i5, reflectance = gt_fit_i5_ref.bin_data(np.mean)
        ax.plot(reflectance, i5, label=f"Cat 4 external,{p}th percentile")

        i5 = list(chain.from_iterable(df_5[f"{p}_EYEWALL_REF_I5"]))
        reflectance = list(chain.from_iterable(df_5[f"{p}_EYEWALL_REF_I4"]))
        gt_fit_i5_ref = GTFit(reflectance, i5)
        i5, reflectance = gt_fit_i5_ref.bin_data(np.mean)
        ax.plot(reflectance, i5, label=f"Cat 5 eyewall,{p}th percentile")
        i5 = list(chain.from_iterable(df_5[f"{p}_EXTERNAL_REF_I5"]))
        reflectance = list(chain.from_iterable(df_5[f"{p}_EXTERNAL_REF_I4"]))
        gt_fit_i5_ref = GTFit(reflectance, i5)
        i5, reflectance = gt_fit_i5_ref.bin_data(np.mean)
        ax.plot(reflectance, i5, label=f"Cat 5 external, {p}th percentile")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("I4 band reflectance")
    ax.set_ylabel("Cloud top temperature (K)")
    ax.legend()
    plt.show()


def plot_binned_i5vsi4r_composite(df, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df[f"{p}_EYEWALL_REF_I5"]))
        reflectance = list(chain.from_iterable(df[f"{p}_EYEWALL_REF_I4"]))
        gt_fit_i5_ref = GTFit(reflectance, i5)
        i5, reflectance = gt_fit_i5_ref.bin_data(np.mean)
        ax.plot(reflectance, i5, label=f"{p}th percentile eyewall")
        i5 = list(chain.from_iterable(df[f"{p}_EXTERNAL_REF_I5"]))
        reflectance = list(chain.from_iterable(df[f"{p}_EXTERNAL_REF_I4"]))
        gt_fit_i5_ref = GTFit(reflectance, i5)
        i5, reflectance = gt_fit_i5_ref.bin_data(np.mean)
        ax.plot(reflectance, i5, label=f"{p}th percentile external")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("I4 band reflectance")
    ax.set_ylabel("Cloud top temperature (K)")
    ax.legend()
    plt.show()


def plot_binned_i5vsi4(df, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df[f"{p}_EYEWALL_I5"]))
        BTD = list(chain.from_iterable(df[f"{p}_EYEWALL_I4"]))
        gt_fit_i5_btd = GTFit(BTD, i5)
        i5, BTD = gt_fit_i5_btd.bin_data(np.mean)
        ax.plot(BTD, i5, label=f"{p}th percentile eyewall")
        i5 = list(chain.from_iterable(df[f"{p}_EXTERNAL_I5"]))
        BTD = list(chain.from_iterable(df[f"{p}_EXTERNAL_I4"]))
        gt_fit_i5_btd = GTFit(BTD, i5)
        i5, BTD = gt_fit_i5_btd.bin_data(np.mean)
        ax.plot(BTD, i5, label=f"{p}th percentile external")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("I4 Brightness Temperature (K)")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    plt.show()


def plot_binned_i5vsBTD(df, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df[f"{p}_EYEWALL_BTD_I5"]))
        BTD = list(chain.from_iterable(df[f"{p}_EYEWALL_BTD"]))
        gt_fit_i5_btd = GTFit(BTD, i5)
        i5, BTD = gt_fit_i5_btd.bin_data(np.mean)
        ax.plot(BTD, i5, label=f"{p}th percentile eyewall")
        i5 = list(chain.from_iterable(df[f"{p}_EXTERNAL_BTD_I5"]))
        BTD = list(chain.from_iterable(df[f"{p}_EXTERNAL_BTD"]))
        gt_fit_i5_btd = GTFit(BTD, i5)
        i5, BTD = gt_fit_i5_btd.bin_data(np.mean)
        ax.plot(BTD, i5, label=f"{p}th percentile external")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_title("Temperature vs Brightness Temperature Difference $T_{I4} - T_{I5}$")
    ax.set_xlabel("Brightness Temperature Difference (K)")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    plt.show()


def plot_binned_i5vsBTD_ratio(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df[f"{p}_EYEWALL_BTD_RATIO_I5"]))
        BTD_RATIO = list(chain.from_iterable(df[f"{p}_EYEWALL_BTD_RATIO"]))
        gt_fit_i5_btd = GTFit(BTD_RATIO, i5)
        i5, BTD_RATIO = gt_fit_i5_btd.bin_data(np.mean)
        ax.plot(BTD_RATIO, i5, label=f"{p}th percentile eyewall")
        i5 = list(chain.from_iterable(df[f"{p}_EXTERNAL_BTD_RATIO_I5"]))
        BTD_RATIO = list(chain.from_iterable(df[f"{p}_EXTERNAL_BTD_RATIO"]))
        gt_fit_i5_btd = GTFit(BTD_RATIO, i5)
        i5, BTD_RATIO = gt_fit_i5_btd.bin_data(np.mean)
        ax.plot(BTD_RATIO, i5, label=f"{p}th percentile external")
    ax.set_title("Cloud top temperature against Brightness temperature difference ratio")
    ax.invert_yaxis()
    ax.set_xlabel("Brightness Temperature Ratio")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    plt.show()


def plot_binned_i5vsi4i5_ratio(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df[f"{p}_EYEWALL_I4_I5_RATIO_I5"]))
        I4_I5_RATIO = list(chain.from_iterable(df[f"{p}_EYEWALL_I4_I5_RATIO"]))
        gt_fit_i5_btd = GTFit(I4_I5_RATIO, i5)
        i5, I4_I5_RATIO = gt_fit_i5_btd.bin_data(np.mean)
        ax.plot(I4_I5_RATIO, i5, label=f"{p}th percentile eyewall")
        i5 = list(chain.from_iterable(df[f"{p}_EXTERNAL_I4_I5_RATIO_I5"]))
        I4_I5_RATIO = list(chain.from_iterable(df[f"{p}_EXTERNAL_I4_I5_RATIO"]))
        gt_fit_i5_btd = GTFit(I4_I5_RATIO, i5)
        i5, I4_I5_RATIO = gt_fit_i5_btd.bin_data(np.mean)
        ax.plot(I4_I5_RATIO, i5, label=f"{p}th percentile external")
    ax.set_title("Cloud top temperature against I4/I5 ")
    ax.invert_yaxis()
    ax.set_xlabel("$T_{I4}/T_{I5}$")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    plt.show()


# def plot_piecewise_composite():
#     fig, ax = plt.subplots(figsize=(8, 8))
#
#     gt_fit_eye = GTFit(eyes_i4, eyes_i5)
#     gt_fit_eye.piecewise_percentile_multiple((5, 50, 95), plot_points=False, setup_axis=False, fig=fig, ax=ax,
#                                              label="Eyewall")
#     enviro_fit_eye = GTFit(environs_i4, environs_i5)
#     enviro_fit_eye.piecewise_percentile_multiple((5, 50, 95), plot_points=False, fig=fig, ax=ax, setup_axis=False,
#                                                  colors=["green", "orange", "blue"], label="External")
#     ax.axhline(-38, c='y', label="$T_{g,homo}$", ls="--")
#     ax.invert_yaxis()
#     ax.invert_xaxis()
#     ax.set_ylabel("Cloud Temperature (C)")
#     ax.set_xlabel("I4 band reflectance (K)")
#     ax.legend()
#     plt.show()


def plot_binned_i5vsre(df, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df[f"{p}_EYEWALL_RE_I5"]))
        re = list(chain.from_iterable(df[f"{p}_EYEWALL_RE"]))
        gt_fit_re = GTFit(re, i5)
        i5, re = gt_fit_re.bin_data(np.mean)
        ax.plot(re, i5, label=f"{p}th percentile eyewall")
        i5 = list(chain.from_iterable(df[f"{p}_EXTERNAL_RE_I5"]))
        re = list(chain.from_iterable(df[f"{p}_EXTERNAL_RE"]))
        gt_fit_re = GTFit(re, i5)
        i5, re = gt_fit_re.bin_data(np.mean)
        ax.plot(re, i5, label=f"{p}th percentile external")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel("Effective Radius (10^-6 m)$")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    plt.show()


def plot_binned_i5vsre_same(df_4, df_5, percentiles, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in percentiles:
        i5 = list(chain.from_iterable(df_4[f"{p}_EYEWALL_RE_I5"]))
        re = list(chain.from_iterable(df_4[f"{p}_EYEWALL_RE"]))
        gt_fit_re = GTFit(re, i5)
        i5, re = gt_fit_re.bin_data(np.mean)
        ax.plot(re, i5, label=f"Cat 4  eyewall, {p}th percentile")
        i5 = list(chain.from_iterable(df_4[f"{p}_EXTERNAL_RE_I5"]))
        re = list(chain.from_iterable(df_4[f"{p}_EXTERNAL_RE"]))
        gt_fit_re = GTFit(re, i5)
        i5, re = gt_fit_re.bin_data(np.mean)
        ax.plot(re, i5, label=f"Cat 4 external, {p}th percentile")
        i5 = list(chain.from_iterable(df_5[f"{p}_EYEWALL_RE_I5"]))
        re = list(chain.from_iterable(df_5[f"{p}_EYEWALL_RE"]))
        gt_fit_re = GTFit(re, i5)
        i5, re = gt_fit_re.bin_data(np.mean)
        ax.plot(re, i5, label=f"Cat 5 eyewall, {p}th percentile")
        i5 = list(chain.from_iterable(df_5[f"{p}_EXTERNAL_RE_I5"]))
        re = list(chain.from_iterable(df_5[f"{p}_EXTERNAL_RE"]))
        gt_fit_re = GTFit(re, i5)
        i5, re = gt_fit_re.bin_data(np.mean)
        ax.plot(re, i5, label=f"Cat 5 external ,{p}th percentile")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel("Effective Radius (10^-6 m)$")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    plt.show()


def overlap_BTD_ratio(*dfs):
    fig, ax = plt.subplots()
    for i, df in enumerate(dfs):
        ax.scatter(df["EYEWALL_I5"], df["EYEWALL_BTD_RATIO"], s=10,
                   label=f"Eyewall {'high wind' if i == 0 else 'low wind'} ")
        # ax.scatter(get_full_column(df, "EXTERNAL_GT"), get_full_column(df, "EXTERNAL_BTD_RATIO"), s=0.5,
        #           label=f"External  {'high wind' if i == 0 else 'low wind' }")
        ax.set_xlabel("I5 Temperature (C)")
        ax.set_ylabel("(I4 - I5) / (I5)")
        ax.legend()
    plt.show()


def plot_ext_eyewall_re_dif(percentiles, df_4, df_5, title=""):
    fig, ax = plt.subplots()
    for percentile in percentiles:
        eyewall_i5 = np.array(list(chain.from_iterable(df_4[f"{percentile}_EYEWALL_RE_I5"])))
        eyewall_reflectance = np.array(list(chain.from_iterable(df_4[f"{percentile}_EYEWALL_RE"])))
        gt_fitter_eyewall = GTFit(eyewall_reflectance, eyewall_i5)
        eyewall_i5, eyewall_reflectance = gt_fitter_eyewall.bin_data(np.mean)
        external_i5 = np.array(list(chain.from_iterable(df_4[f"{percentile}_EXTERNAL_RE_I5"])))
        external_reflectance = np.array(list(chain.from_iterable(df_4[f"{percentile}_EXTERNAL_RE"])))
        gt_fitter_external = GTFit(external_reflectance, external_i5)
        external_i5, external_reflectance = gt_fitter_external.bin_data(np.mean)
        delta_ir = eyewall_reflectance - external_reflectance
        ax.plot(delta_ir, external_i5, label=f"{percentile}th percentile difference, Category 4")
        eyewall_i5 = np.array(list(chain.from_iterable(df_5[f"{percentile}_EYEWALL_RE_I5"])))
        eyewall_reflectance = np.array(list(chain.from_iterable(df_5[f"{percentile}_EYEWALL_RE"])))
        gt_fitter_eyewall = GTFit(eyewall_reflectance, eyewall_i5)
        eyewall_i5, eyewall_reflectance = gt_fitter_eyewall.bin_data(np.mean)
        external_i5 = np.array(list(chain.from_iterable(df_5[f"{percentile}_EXTERNAL_RE_I5"])))
        external_reflectance = np.array(list(chain.from_iterable(df_5[f"{percentile}_EXTERNAL_RE"])))
        gt_fitter_external = GTFit(external_reflectance, external_i5)
        external_i5, external_reflectance = gt_fitter_external.bin_data(np.mean)
        delta_ir = eyewall_reflectance - external_reflectance
        ax.plot(delta_ir, external_i5, label=f"{percentile}th percentile difference, Category 5")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Temperature (K)")
    ax.set_xlabel("Eyewall $r_e$ - External $r_e$  ($\mu m$)")
    plt.show()


def plot_ext_eyewall_dif(percentiles, df_4, df_5, title=""):
    fig, ax = plt.subplots()
    for percentile in percentiles:
        eyewall_i5 = np.array(list(chain.from_iterable(df_4[f"{percentile}_EYEWALL_REF_I5"])))
        eyewall_reflectance = np.array(list(chain.from_iterable(df_4[f"{percentile}_EYEWALL_REF_I4"])))
        gt_fitter_eyewall = GTFit(eyewall_reflectance, eyewall_i5)
        eyewall_i5, eyewall_reflectance = gt_fitter_eyewall.bin_data(np.mean)
        external_i5 = np.array(list(chain.from_iterable(df_4[f"{percentile}_EXTERNAL_REF_I5"])))
        external_reflectance = np.array(list(chain.from_iterable(df_4[f"{percentile}_EXTERNAL_REF_I4"])))
        gt_fitter_external = GTFit(external_reflectance, external_i5)
        external_i5, external_reflectance = gt_fitter_external.bin_data(np.mean)
        delta_ir = eyewall_reflectance - external_reflectance
        ax.plot(delta_ir, external_i5, label=f"{percentile}th percentile difference, Category 4")
        eyewall_i5 = np.array(list(chain.from_iterable(df_5[f"{percentile}_EYEWALL_REF_I5"])))
        eyewall_reflectance = np.array(list(chain.from_iterable(df_5[f"{percentile}_EYEWALL_REF_I4"])))
        gt_fitter_eyewall = GTFit(eyewall_reflectance, eyewall_i5)
        eyewall_i5, eyewall_reflectance = gt_fitter_eyewall.bin_data(np.mean)
        external_i5 = np.array(list(chain.from_iterable(df_5[f"{percentile}_EXTERNAL_REF_I5"])))
        external_reflectance = np.array(list(chain.from_iterable(df_5[f"{percentile}_EXTERNAL_REF_I4"])))
        gt_fitter_external = GTFit(external_reflectance, external_i5)
        external_i5, external_reflectance = gt_fitter_external.bin_data(np.mean)
        delta_ir = eyewall_reflectance - external_reflectance
        ax.plot(delta_ir, external_i5, label=f"{percentile}th percentile difference, Category 5")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.legend()
    ax.set_ylabel("Temperature (K)")
    ax.set_xlabel("Eyewall reflectance - External reflectance")
    plt.show()


def vs_vmax(df, percentile=95, wind_time=0, fig=None, ax=None, label=""):
    vmax = []
    delta_ref = []
    for idx, row in df.iterrows():
        external_temp, external_reflectance = GTFit(row[f"{percentile}_EXTERNAL_REF_I4"],
                                                    row[f"{percentile}_EXTERNAL_REF_I5"]).bin_data(np.mean,
                                                                                                   custom_range=(
                                                                                                       260, 280))
        eyewall_temp, eyewall_reflectance, = GTFit(row[f"{percentile}_EYEWALL_REF_I4"],
                                                   row[f"{percentile}_EYEWALL_REF_I5"]).bin_data(np.mean,
                                                                                                 custom_range=(
                                                                                                     260, 280))
        min_len = min(len(eyewall_reflectance), len(external_reflectance))
        delta_reflectance = eyewall_reflectance[:min_len] - external_reflectance[:min_len]
        if np.mean(delta_reflectance) < 0.025 and row["USA_SSHS"] == 5:
            continue
        if np.isnan(np.mean(delta_reflectance)):
            continue
        if wind_time == 0:
            vmax.append(row["USA_WIND"])
        else:
            steps = wind_time // 3
            idx = int(row["START_IDX"] + steps)
            if np.isnan(best_track_df.iloc[idx]["USA_WIND"]):
                continue
            vmax.append(best_track_df.iloc[idx]["USA_WIND"])
        delta_ref.append(np.mean(delta_reflectance))
    print(sps.pearsonr(delta_ref, vmax))
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.scatter(delta_ref, vmax, label=label)
    ax.set_xlabel("Eyewall reflectance - external reflectance")
    ax.set_ylabel("$V_{max}$ (kts)")


def changeI5mask(ci: CycloneImageFast, new_low=None, new_high=None):
    ci.unmask(recalculate=False)

    ci.mask(low=new_low, high=new_high, calculate=True)
    ci.eye().i4_reflectance = ci.eye().calc.reflectance_from_tbs(ci.eye().zenith, ci.eye().i4, ci.eye().i5)
    for c in ci.cells:
        c.i4_reflectance = c.calc.reflectance_from_tbs(c.zenith, c.i4, c.i5)


if __name__ == "__main__":
    files = glob.glob(os.path.join(CACHE_DIRECTORY, "**.gzp"))
    FILE = r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\new2_gts.gzp"
    # percentiles = (5, 50, 95)
    if os.path.isfile(FILE):
        cyclone_df = pd.read_pickle(FILE, compression="gzip")
    else:
        cyclone_df = pd.DataFrame()
        for f in tqdm(files):
            try:
                ci = CycloneImageFast.from_gzp(f)
                if not hasattr(ci, "cells"):
                    ci.generate_environmental()
                    ci.pickle()
                else:
                    print(len(ci.cells))
                # EYE
                try:
                    ci.eye().gt, ci.eye().gt_i4 = ci.eye().glaciation_temperature_mean()
                except ValueError:
                    continue
                # ci.plot_eyewall_against_external()
                # ci.plot_eyewall_against_ext_ref()

                if ci.eye().good_gt:
                    ci_dict = ci.metadata
                    ci_dict["GT_EYEWALL"] = ci.eye().gt.value
                    ci_dict["GT_EYEWALL_ERR"] = ci.eye().gt.error
                    ci_dict["EXTERNAL_GT"], ci_dict["EXTERNAL_GT_ERR"] = ci.get_environmental_gts()
                    # # ci_dict["I4_REFLECTANCE"] = ci.eye().i4_reflectance
                    #
                    # # re = ci.eye().re(True).flatten()
                    # # if len(re[~np.isnan(re)]) < 25:
                    # #     continue
                    # #
                    # # gt_fit_re = GTFit(re, ci.eye().i5.flatten()[~np.isnan(re)])
                    # gt_fit_eye_i4r = GTFit(ci.eye().i4_reflectance_flat, ci.eye().i5_flat)
                    # gt_fit_eye = GTFit(ci.eye().i4_flat, ci.eye().i5_flat)
                    # gt_fit_eye_BTD = GTFit(ci.eye().BTD, ci.eye().i5_flat)
                    # gt_fit_eye_BTD_ratio = GTFit(ci.eye().BTD_ratio, ci.eye().i5_flat)
                    # gt_fit_eye_i4i5ratio = GTFit(ci.eye().i4i5ratio, ci.eye().i5_flat)
                    # for p in percentiles:
                    #     ci_dict[f"{p}_EYEWALL_REF_I5"], ci_dict[f"{p}_EYEWALL_REF_I4"] = gt_fit_eye_i4r.bin_data(
                    #         np.percentile, 1, bin_func_args=(p,))
                    #     ci_dict[f"{p}_EYEWALL_I5"], ci_dict[f"{p}_EYEWALL_I4"] = gt_fit_eye.bin_data(
                    #         np.percentile, 1, bin_func_args=(p,))
                    #     ci_dict[f"{p}_EYEWALL_BTD_I5"], ci_dict[f"{p}_EYEWALL_BTD"] = gt_fit_eye_BTD.bin_data(
                    #         np.percentile, 1, bin_func_args=(p,))
                    #     ci_dict[f"{p}_EYEWALL_BTD_RATIO_I5"], ci_dict[
                    #         f"{p}_EYEWALL_BTD_RATIO"] = gt_fit_eye_BTD_ratio.bin_data(np.percentile, 1,
                    #                                                                   bin_func_args=(p,))
                    #     ci_dict[f"{p}_EYEWALL_I4_I5_RATIO_I5"], ci_dict[
                    #         f"{p}_EYEWALL_I4_I5_RATIO"] = gt_fit_eye_i4i5ratio.bin_data(np.percentile, 1,
                    #                                                                     bin_func_args=(p,))
                    #     # ci_dict[f"{p}_EYEWALL_RE_I5"], ci_dict[
                    #     #     f"{p}_EYEWALL_RE"] = gt_fit_re.bin_data(np.percentile, 1, bin_func_args=(p,))
                    #
                    #     e_i4, e_btd, e_btd_r, e_i4i5, e_i4r = [], [], [], [], []
                    #     e_i5, e_btd_i5, e_btd_r_i5, e_i4i5_i5, e_i4ri5 = [], [], [], [], []
                    #     # e_re, e_re_i5 = [], []
                    #     for c in ci.cells:
                    #         try:
                    #             gt_fit_ext = GTFit(c.i4_flat, c.i5_flat)
                    #             gt_fit_ext_BTD = GTFit(c.BTD, c.i5_flat)
                    #             gt_fit_ext_BTD_ratio = GTFit(c.BTD_ratio, c.i5_flat)
                    #             gt_fit_ext_i4_i5 = GTFit(c.i4i5ratio, c.i5_flat)
                    #             gt_fit_ext_i4r = GTFit(c.i4_reflectance_flat, c.i5_flat)
                    #             # re_ext = c.re(True).flatten()
                    #             # if len(re_ext) < 25:
                    #             #     continue
                    #             # gt_fit_ext_re = GTFit(re_ext, c.i5.flatten()[~np.isnan(re_ext)])
                    #             # i5, re_ext = gt_fit_ext_re.bin_data(np.percentile, 1, bin_func_args=(p,))
                    #             # e_re.extend(re_ext)
                    #             # e_re_i5.extend(i5)
                    #             #
                    #             i5, i4r = gt_fit_ext_i4r.bin_data(np.percentile, 1, bin_func_args=(p,))
                    #             e_i4r.extend(i4r)
                    #             e_i4ri5.extend(i5)
                    #
                    #             i5, i4 = gt_fit_ext.bin_data(np.percentile, 1, bin_func_args=(p,))
                    #             e_i5.extend(i5)
                    #             e_i4.extend(i4)
                    #             i5, BTD = gt_fit_ext_BTD.bin_data(np.percentile, 1, bin_func_args=(p,))
                    #             e_btd.extend(BTD)
                    #             e_btd_i5.extend(i5)
                    #             i5, BTD_ratio = gt_fit_ext_BTD_ratio.bin_data(np.percentile, 1, bin_func_args=(p,))
                    #             e_btd_r.extend(BTD_ratio)
                    #             e_btd_r_i5.extend(i5)
                    #             i5, i4i5_ratio = gt_fit_ext_i4_i5.bin_data(np.percentile, 1, bin_func_args=(p,))
                    #             e_i4i5_i5.extend(i5)
                    #             e_i4i5.extend(i4i5_ratio)
                    #         except ValueError:
                    #             continue
                    #
                    #     ci_dict[f"{p}_EXTERNAL_I5"] = e_i5
                    #     ci_dict[f"{p}_EXTERNAL_I4"] = e_i4
                    #     ci_dict[f"{p}_EXTERNAL_BTD"] = e_btd
                    #     ci_dict[f"{p}_EXTERNAL_BTD_I5"] = e_btd_i5
                    #     ci_dict[f"{p}_EXTERNAL_BTD_RATIO"] = e_btd_r
                    #     ci_dict[f"{p}_EXTERNAL_BTD_RATIO_I5"] = e_btd_r_i5
                    #     ci_dict[f"{p}_EXTERNAL_I4_I5_RATIO"] = e_i4i5
                    #     ci_dict[f"{p}_EXTERNAL_I4_I5_RATIO_I5"] = e_i4i5_i5
                    #     ci_dict[f"{p}_EXTERNAL_REF_I5"] = e_i4ri5
                    #     ci_dict[f"{p}_EXTERNAL_REF_I4"] = e_i4r

                    # ci_dict[f"{p}_EXTERNAL_RE"] = e_re
                    # ci_dict[f"{p}_EXTERNAL_RE_I5"] = e_re_i5

                    cyclone_df = cyclone_df.append(ci_dict, ignore_index=True)
            except AssertionError:
                import traceback

                traceback.print_exc()
        cyclone_df.to_pickle(FILE, compression="gzip")
    eyewall_hist(cyclone_df)
    # i5_e, i4_e = bin_mean(cyclone_df, "95_EXTERNAL_I5", "95_EXTERNAL_I4")
    # i5_ewall, i4_ewall = bin_mean(cyclone_df, "95_EYEWALL_I5", "95_EYEWALL_I4")
    # delta_frac = (i4_ewall) / i4_e
    # print(np.mean(delta_frac))
    # print(sem(delta_frac))
    #
    # cat_4_df = cyclone_df[cyclone_df["USA_SSHS"] == 4]
    # cat_5_df = cyclone_df[cyclone_df["USA_SSHS"] == 5]
    # #
    # plot_binned_i5vsre_same(cat_4_df,cat_5_df, percentiles=percentiles, title=f"Temperature vs Effective Radius")
    # plot_binned_i5vsre(cat_5_df, title="Cateogry 5: Temperature vs Effective Radius")
    # fig,ax = plt.subplots()
    # vs_vmax(cat_4_df, 95, wind_time=0,fig=fig,ax = ax,label="Category 4")
    # vs_vmax(cat_5_df, 95, wind_time=0,fig=fig,ax=ax,label="Category 5")
    # ax.legend()
    # plt.show()
    # plot_ext_vs_eyedif(95, cyclone_df)
    # plot_binned_i5vsi4(cyclone_df)
    # plot_binned_i5vsBTD(cyclone_df)
    # plot_binned_i5vsBTD_ratio(cyclone_df)
    # plot_binned_i5vsi4i5_ratio(cyclone_df)
