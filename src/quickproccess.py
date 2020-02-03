import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

files = [r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\IRMA\2017-09-05 17-00\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\IRMA\2017-09-06 16-42\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\IRMA\2017-09-06 18-24\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\IRMA\2017-09-07 18-06\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\IRMA\2017-09-08 17-48\out.json",
         # r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\LEKIMA\2019-08-08 04-06\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\WALAKA\2018-10-01 23-18\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\WALAKA\2018-10-02 00-54\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\WALAKA\2018-10-03 00-36\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\YUTU\2018-10-24 04-06\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\YUTU\2018-10-25 03-48\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\YUTU\2018-10-26 03-30\out.json",
         r"C:\Users\Robert\PycharmProjects\TropicalCyclonesMSci\out\YUTU\2018-10-27 03-12\out.json"]

lb = []
rb = []
lf = []
rf = []
avg = []
var = []
for file in files:
    with open(file) as f:
        obj = json.load(f)
        l = []
        if not np.isnan(obj["LB"]):
            lb.append(obj["LB"])
            l.append(obj["LB"])
        if not np.isnan(obj["RB"]):
            rb.append(obj["RB"])
            l.append(obj["RB"])
        if not np.isnan(obj["RF"]):
            rf.append(obj["RF"])
            l.append(obj["RF"])
        if not np.isnan(obj["LF"]):
            lf.append(obj["LF"])
            l.append(obj["LF"])
        l = np.array(l)
        var.append(l.std())


lb = np.array(lb)
rb = np.array(rb)
rf = np.array(rf)
lf = np.array(lf)

print(f"Mean LF:{np.nanmean(lf)} std:{sem(lf)}")
print(f"Mean RF:{np.nanmean(rf)} std:{sem(rf)}")
print(f"Mean LB:{np.nanmean(lb)} std:{sem(lb)}")
print(f"Mean  RB:{np.nanmean(rb)} std:{sem(rb)}")

print(f"Avg std:{np.nanmean(var)}")