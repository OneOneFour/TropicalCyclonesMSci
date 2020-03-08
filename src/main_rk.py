from time import strptime

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# Top level import required in order for pickle class to be loaded into the __main__ namespace
from BestTrack import *

register_matplotlib_converters()

font = {'family': 'normal',
        'weight': 'bold',
        'size': 15}

matplotlib.rc("font", **font)


def plt_time_series(series):
    fig, ax = plt.subplots()
    formatter = mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
    dates = [strptime(s["ISO_TIME"], "%Y-%m-%d %H:%M:%S") for s in series]
    ax.plot(dates, [s["EYE"] for s in series], label="Eye", linestyle="solid")
    ax.plot(dates, [s["RB"] for s in series], label="Right Back", linestyle="solid")
    ax.plot(dates, [s["LB"] for s in series], label="Left Back", linestyle="solid")
    ax.plot(dates, [s["RF"] for s in series], label="Right Forward", linestyle="solid")
    ax.plot(dates, [s["LF"] for s in series], label="Left Forward", linestyle="solid")
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ## ALL
    all_cyclones_since(2017, 8, 1, per_cyclone=lambda x: x.auto_gt_cycle())
    # eyes = get_cyclone_eye_name_image("GENEVIEVE", 2014, max_len=5)
    # hato = get_cyclone_by_name("HATO",2017,per_cyclone=lambda x: x.auto_gt_cycle())
    # get_cyclone_by_name("IRMA", 2017, per_cyclone=lambda x: x.auto_gt_cycle())
