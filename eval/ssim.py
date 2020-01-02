import skimage
from skimage import io, transform
from skimage.measure import compare_ssim
import pdb

gt = io.imread("../image/117-gt.png")
# print(gt)
print(gt.shape)
# io.imshow(gt[150:608, 250:676, :])
# io.show()

rnn = io.imread("../image/117-rnn.png")
print(rnn.shape)
lerp = io.imread("../image/117-lerp.png")
tsr_tvd = io.imread("../image/117-tsr-tvd.png")

# gt_sm = transform.resize(gt, (256, 256))
# lerp_sm = transform.resize(lerp, (256, 256))
# rnn_sm = transform.resize(rnn, (256, 256))
# tsr_tvd_sm = transform.resize(tsr_tvd, (256, 256))
# io.imsave("../image/117-gt-sm.png", gt_sm)
# io.imsave("../image/117-rnn-sm.png", rnn_sm)
# io.imsave("../image/117-lerp-sm.png", lerp_sm)
# io.imsave("../image/117-tsr-tvd-sm.png", tsr_tvd_sm)
# pdb.set_trace()

gt_sub = gt[150:608, 250:676, :]
rnn_sub = rnn[150:608, 250:676, :]
lerp_sub = lerp[150:608, 250:676, :]
tsr_tvd_sub = tsr_tvd[150:608, 250:676, :]
io.imsave("../image/117-gt-sub.png", gt_sub)
io.imsave("../image/117-rnn-sub.png", rnn_sub)
io.imsave("../image/117-lerp-sub.png", lerp_sub)
io.imsave("../image/117-tsr-tvd-sub.png", tsr_tvd_sub)


ssim_rnn = compare_ssim(gt_sub, rnn_sub, data_range=255., multichannel=True)
ssim_lerp = compare_ssim(gt_sub, lerp_sub, data_range=255., multichannel=True)
ssim_tsr_tvd = compare_ssim(gt_sub, tsr_tvd_sub, data_range=255., multichannel=True)
print("SSIM_LERP: {}".format(ssim_lerp))
print("SSIM_RNN: {}".format(ssim_rnn))
print("SSIM_TSR_TVD: {}".format(ssim_tsr_tvd))

