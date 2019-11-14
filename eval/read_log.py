import pdb
import numpy as np

f = open("psnr.log")
line = f.readline()
test_start = 50
infer_step = 3

offset1 = []
offset2 = []

file_count = 0
while line:
    if line[-1] == '\n':
        line = line[:-1]
    timestep = int(line[line.rfind("_")+1:line.find(",")])
    psnr = float(line[line.rfind(" ")+1:])

    offset = (timestep - test_start) % (infer_step+1)
    if offset == 1 or offset == 3:
        offset1.append(psnr)
    if offset == 2:
        offset2.append(psnr)
    # pdb.set_trace()
    line = f.readline()
print("Avg1 {}".format(np.array(offset1).mean()))
print("Avg2 {}".format(np.array(offset2).mean()))


