import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from BestTrack import get_cyclone_by_name


def plt_time_series(series):
    dates = [mdates.date2num(s["ISO_TIME"]) for s in series]
    plt.plot_date(x=dates, x_date=True, y=[s["EYE"] for s in series], label="Eye")
    plt.plot_date(x=dates, x_date=True, y=[s["RB"] for s in series], label="Right Back")
    plt.plot_date(x=dates, x_date=True, y=[s["LB"] for s in series], label="Left Back")
    plt.plot_date(x=dates, x_date=True, y=[s["RF"] for s in series], label="Right Forward")
    plt.plot_date(x=dates, x_date=True, y=[s["LF"] for s in series], label="Left Forward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ## NA
    irma_series = get_cyclone_by_name("IRMA", 2017,max_len=4, per_cyclone=lambda x: x.auto_gt_cycle())
    plt_time_series(irma_series)
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
