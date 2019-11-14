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
    parser.add_argument("--infering-step", type=int, default=7,
                        help="in the infering phase, the number of intermediate volumes")
    return parser.parse_args()

def main(args):
    zSize, ySize, xSize = 120, 720, 480
    volume_size_r = 1. / zSize / ySize / xSize

    gt_root = os.path.join(args.root, "exavisData", "combustion")
    psnrs = 0
    file_count = 0
    for i in range(args.test_start, args.test_end+1):
        if i % (args.infering_step+1) != 2:
            idx = ("%04d" % i)
            gt_filepath = os.path.join(gt_root, "jet_" + idx, "jet_mixfrac_" + idx + ".dat")
            gt = volume_loader(gt_filepath, zSize, ySize, xSize)
            pred_filepath = os.path.join(args.root, "save_pred", "jet_mixfrac_" + idx + ".raw")
            pred = volume_loader(pred_filepath, zSize, ySize, xSize)
            mse = np.sum(np.power(gt - pred, 2.)) * volume_size_r
            diff = gt.max() - gt.min()
            psnr = 20. * np.log10(diff) - 10. * np.log10(mse)
            print("jet_mixfrac_{}, PSNR {}".format(idx, psnr))
            psnrs += psnr
            file_count += 1
    print("Average PSNR: {}".format(psnrs / file_count))

if __name__ == "__main__":
    main(parse_args())




