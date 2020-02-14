from time import strptime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from BestTrack import get_cyclone_by_name

register_matplotlib_converters()


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
    # all_cyclones_since(2014, 12, 1, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("DORIAN", 2019, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("IRMA", 2017, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("GENEVIEVE", 2014, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("HELLEN", 2014, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("IAN", 2014, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("JELAWAT", 2012, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("LEKIMA", 2013, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("NEOGURI", 2014, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("RAMMASUN", 2014, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("USAGI", 2013, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("WALAKA", 2018, per_cyclone=lambda x: x.auto_gt_cycle())
    get_cyclone_by_name("YUYU", 2018, per_cyclone=lambda x: x.auto_gt_cycle())
