import matplotlib
from pandas.plotting import register_matplotlib_converters

# Top level import required in order for pickle class to be loaded into the __main__ namespace
from BestTrack import *

register_matplotlib_converters()

font = {'family': 'normal',
        'weight': 'bold',
        'size': 15}

matplotlib.rc("font", **font)

if __name__ == "__main__":
    all_cyclones_since(2018,8,1)
    # eyes = get_cyclone_eye_name_image("GENEVIEVE", 2014, max_len=5)
    # hato = get_cyclone_by_name("HATO",2017,per_cyclone=lambda x: x.auto_gt_cycle())
    # get_cyclone_by_name_date("IRMA", datetime(2017, 9, 8, 0, 0, 0), datetime(2017, 9, 8, 23, 59, 59),
    # get_cyclone_by_name("IRMA", 2017, per_cyclone=lambda x: x.get_environmental_gt())
