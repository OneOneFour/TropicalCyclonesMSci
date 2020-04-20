import matplotlib.pyplot as plt
import numpy as np
from GTFit import simple_piecewise

if __name__ == "__main__":
    fig, ax = plt.subplots()
    sample_gt = 235
    i4_turningpoint = 3
    a, b = 0.1,0.5
    t = np.arange(210, 270, 1.0)
    i4_r = simple_piecewise(t, sample_gt, i4_turningpoint, a, b)
    ax.plot(i4_r, t, label="Model")
    ax.set_ylabel("Temperature (K)")
    ax.axhspan(ymin=210,ymax=sample_gt,alpha=0.3,facecolor="yellow")
    ax.axhspan(ymin=sample_gt,ymax=270,alpha=0.3,facecolor="green")
    ax.axhline(sample_gt,ls="--",c="r")
    right,left = ax.get_xlim()
    ax.annotate("Glaciation Temperature",xy=(left - 0.25,sample_gt-0.3 ),ha="left",fontsize=10,weight="bold",color="darkred")
    ax.annotate("Fully Glaciated Region",xy=(16,0.5*(210+sample_gt) + 3),fontsize=15,ha="center", weight="bold", color="darkgoldenrod")
    ax.annotate("Mixed Phase Region", xy=(16, 0.5 * (270+ sample_gt) - 3), fontsize=15, ha="center", weight="bold",
                color="darkgreen")
    ## X axis labels
    ax.annotate("Increasing $r_e$\nDecreasing $\\rho$",xy=(right,274),xytext=(right+3,274),
                arrowprops=dict(arrowstyle='->'),annotation_clip=False,ha="center",va="center")

    ax.annotate("Decreasing $r_e$\nIncreasing $\\rho$", xy=(left, 274), xytext=(left - 3, 274),
                arrowprops=dict(arrowstyle='->'), annotation_clip=False, ha="center", va="center")

    ax.set_ylim((210,270))
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.show()
