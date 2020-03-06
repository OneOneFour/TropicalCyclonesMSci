import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

df = pd.read_csv("out.csv")
# df = df.loc[df.aod != "--"]
# df = df.astype({"aod": "float"})
fig, ax = plt.subplots()
ax.hist(df["gt"], bins=np.arange(-44, -10, 2))
ax.set_xlabel("Glaciation Temperature (C)")
ax.set_ylabel("Frequency")
plt.show()
#
# fig, ax = plt.subplots()
# ax.hist(df["i4"], bins=15)
# ax.set_xlabel("I4 band reflectance (K)")
# ax.set_ylabel("Frequency")
#
# fig, ax = plt.subplots()
# na_points = df.loc[df["basin"] == "NA"]
# non_na_points = df.loc[df["basin"] != "NA"]
# ax.scatter(na_points["gt"].values, na_points["aod"].values, label="NA")
# ax.scatter(non_na_points["gt"].values, non_na_points["aod"].values, label="Not NA")
# plt.legend()
# ax.set_xlabel("Glaciation Temperature (K)")
# ax.set_ylabel("Aerosol Optical Depth")
#
# plt.show()
#
# print(stats.pearsonr(df["gt"], df["i4"]))
# print(stats.pearsonr(df["gt"], df["aod"]))
