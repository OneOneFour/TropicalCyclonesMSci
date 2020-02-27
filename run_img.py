from CycloneImage import CycloneImage
import numpy as np

if __name__ == "__main__":
    path = input("Enter pickle path")
    ci = CycloneImage.load(path)
    valid = [snap for snap in ci.rects if not np.isnan(snap.gt_piece_percentile(plot=False)[0])]
    valid[2].unmask_array()
    valid[2].gt_piece_all()