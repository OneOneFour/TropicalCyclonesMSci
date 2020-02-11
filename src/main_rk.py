from time import strptime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from BestTrack import get_cyclone_by_name
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plt_time_series(series):
    fig,ax = plt.subplots()
    formatter = mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
    dates = [strptime(s["ISO_TIME"],"%Y-%m-%d %H:%M:%S") for s in series]
    ax.plot(dates,  [s["EYE"] for s in series], label="Eye", linestyle="solid")
    ax.plot(dates,  [s["RB"] for s in series], label="Right Back", linestyle="solid")
    ax.plot(dates, [s["LB"] for s in series], label="Left Back", linestyle="solid")
    ax.plot(dates,  [s["RF"] for s in series], label="Right Forward", linestyle="solid")
    ax.plot(dates,  [s["LF"] for s in series], label="Left Forward", linestyle="solid")
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ## NA
    irma_series = get_cyclone_by_name("GENEVIEVE", 2014, per_cyclone=lambda x: x.auto_gt_cycle())
    # walaka = get_cyclone_by_name_date("WALAKA", datetime(year=2018, month=10, day=2, hour=0, minute=0),
    #                                    datetime(year=2018, month=10, day=2, hour=6, minute=0))
    # walaka.grid_data(7,7,2*96,2*96)
    # michael = get_cyclone_by_name("MICHAEL", 2018,max_len=1)
    # dorian = get_cyclone_by_name("DORIAN", 2019,max_len=1)
    # # EP
    # patricia = get_cyclone_by_name("PATRICIA", 2015,max_len=1)
    # # WP
    # yutu = get_cyclone_by_name("YUTU", 2018,max_len=1)
    # vongfong = get_cyclone_by_name("VONGFONG", 2014,max_len=1)
    # haiyan = get_cyclone_by_name("HAIYAN", 2013,max_len=1)
    # r = cis[0].draw_rectangle((16.5, -55.283), 250000, 250000)
    # r_2 = cis[0].draw_rectangle(((16.13, -61.9)), 100000, 250000)
