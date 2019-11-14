import os
import pdb

time_start = 50
time_end = 122
old_infer_step = 3
new_infer_step = 7

for i in range(time_start, time_end, new_infer_step+1):
    range_start, range_end = i, i+new_infer_step+1

    target_file = "volume_test_list_" + str(range_start) + "-" + str(range_end) + ".txt"
    head_file = "volume_test_list_" + str(range_start) + "-" + str(range_start+old_infer_step+1) + ".txt"
    tail_file = "volume_test_list_" + str(range_end-old_infer_step-1) + "-" + str(range_end) + ".txt"
    pdb.set_trace()

    cmd1 = "head -9451 " + head_file + " > " + target_file
    cmd2 = "tail -9450 " + tail_file + " >> " + target_file
    os.system(cmd1)
    os.system(cmd2)

