fo = open("volume_train_list.txt", "w")
xSize, ySize, zSize = 480, 720, 120
fo.write("{} {} {}\n".format(xSize, ySize, zSize))
for i in range(49):
    if i+1 < 10:
        fo.write("jet_000{}/jet_mixfrac_000{}.dat\n".format(i+1, i+1))
    else:
        fo.write("jet_00{}/jet_mixfrac_00{}.dat\n".format(i+1, i+1))

fo = open("volume_test_list.txt", "w")
fo.write("{} {} {}\n".format(xSize, ySize, zSize))
for i in range(49, 122):
    if i+1 < 100:
        fo.write("jet_00{}/jet_mixfrac_00{}.dat\n".format(i + 1, i + 1))
    else:
        fo.write("jet_0{}/jet_mixfrac_0{}.dat\n".format(i + 1, i + 1))