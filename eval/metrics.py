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
    parser.add_argument("--lerp", action="store_true", default=False,
                        help="set the lerp mode")
    return parser.parse_args()

def main(args):
    zSize, ySize, xSize = 120, 720, 480
    volume_size_r = 1. / zSize / ySize / xSize

    gt_root = os.path.join(args.root, "exavisData", "combustion")
    psnrs = 0
    file_count = 0
    if args.lerp:
        start_idx = ("%04d" % args.test_start)
        gt_start_filepath = os.path.join(gt_root, "jet_" + start_idx, "jet_mixfrac_" + start_idx + ".dat")
        gt_start = volume_loader(gt_start_filepath, zSize, ySize, xSize)
        end_idx = ("%04d" % args.test_end)
        gt_end_filepath = os.path.join(gt_root, "jet_" + end_idx, "jet_mixfrac_" + end_idx + ".dat")
        gt_end = volume_loader(gt_end_filepath, zSize, ySize, xSize)

    for i in range(args.test_start, args.test_end+1):
        if (i - args.test_start) % (args.infering_step+1) != 0:
            idx = ("%04d" % i)
            gt_filepath = os.path.join(gt_root, "jet_" + idx, "jet_mixfrac_" + idx + ".dat")
            gt = volume_loader(gt_filepath, zSize, ySize, xSize)

            if args.lerp:
                offset = i - args.test_start
                interval = args.test_end - args.test_start
                pred = (1-offset/interval) * gt_start + offset/interval * gt_end
            else:
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




