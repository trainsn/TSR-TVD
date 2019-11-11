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
    parser.add_argument("--time-start", type=int, default=50,
                        help="starting key timestep")
    parser.add_argument("--time-end ", type=int. default=)
    return parser.parse_args()

def main(args):
    zSize, ySize, xSize = 120, 720, 480
    volume_size_r = 1. / zSize / ySize / xSize

    gt_root = os.path.join(args.root, "exavisData", "combustion")
    test_start = 50
    test_end = 54
    infering_step = 3
    psnrs = 0
    file_count = 0
    for i in range(test_start, test_end+1):
        if i % (infering_step+1) != 2:
            idx = ("%04d" % i)
            gt_filepath = os.path.join(gt_root, "jet_" + idx, "jet_mixfrac_" + idx + ".dat" )
            gt = volume_loader(gt_filepath, zSize, ySize, xSize)
            pred_filepath = os.path.join(args.root, "save_pred", "jet_mixfrac_" + idx + ".raw")
            pred = volume_loader(pred_filepath, zSize, ySize, xSize)
            mse = np.sum(np.power(gt - pred, 2.)) * volume_size_r
            diff = gt.max() - gt.min()
            psnr = 20. * np.log10(diff) - 10. * np.log10(mse)
            print("jet_mixfrac_{}, PSNR {}".format(idx, psnr))
            psnrs += psnr
            file_count += 1
    print("Average PSNR: {}", psnrs)

if __name__ == "__main__":
    main(parse_args())







