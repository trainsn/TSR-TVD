import os
import sys
import argparse
import numpy as np
import pdb
sys.path.append("../datasets")
from trainDataset import volume_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")
    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--test-start", type=int, default=50,
                        help="starting key timestep")
    parser.add_argument("--test-end", type=int, default=66,
                        help="ending key timestep")
    parser.add_argument("--infering-step", type=int, default=9,
                        help="in the infering phase, the number of intermediate volumes")
    return parser.parse_args()

def main(args):
    zSize, ySize, xSize = 120, 720, 480

    gt_root = os.path.join(args.root, "exavisData", "combustion")

    start_idx = ("%04d" % args.test_start)
    gt_start_filepath = os.path.join(gt_root, "jet_" + start_idx, "jet_mixfrac_" + start_idx + ".dat")
    gt_start = volume_loader(gt_start_filepath, zSize, ySize, xSize)
    end_idx = ("%04d" % args.test_end)
    gt_end_filepath = os.path.join(gt_root, "jet_" + end_idx, "jet_mixfrac_" + end_idx + ".dat")
    gt_end = volume_loader(gt_end_filepath, zSize, ySize, xSize)

    for i in range(args.test_start+1, args.test_end):
        offset = i - args.test_start
        interval = args.test_end - args.test_start
        pred = (1 - offset / interval) * gt_start + offset / interval * gt_end
        pred = pred.astype(np.float32)
        volume_name = "jet_mixfrac_" + ("%04d" % i) + '.raw'
        pred.tofile(os.path.join(args.root, "save_lerp", volume_name))

if __name__ == "__main__":
    main(parse_args())