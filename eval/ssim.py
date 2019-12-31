import skimage
from skimage import io, transform
from skimage.measure import compare_ssim
import pdb

gt = io.imread("../image/vortex-gt.png")
# print(gt)
# print(gt.shape)
# io.imshow(gt[150:608, 250:676, :])
# io.show()

rnn = io.imread("../image/vortex-rnn.png")
lerp = io.imread("../image/vortex-lerp.png")
tsr_tvd = io.imread("../image/vortex-tsr-tvd.png")

# gt_sm = transform.resize(gt, (256, 256))
# lerp_sm = transform.resize(lerp, (256, 256))
# rnn_sm = transform.resize(rnn, (256, 256))
# tsr_tvd_sm = transform.resize(tsr_tvd, (256, 256))
# io.imsave("../image/vortex-gt-sm.png", gt_sm)
# io.imsave("../image/vortex-rnn-sm.png", rnn_sm)
# io.imsave("../image/vortex-lerp-sm.png", lerp_sm)
# io.imsave("../image/vortex-tsr-tvd-sm.png", tsr_tvd_sm)
# pdb.set_trace()

# gt_sub = gt[150:608, 250:676, :]
# rnn_sub = rnn[150:608, 250:676, :]
# lerp_sub = lerp[150:608, 250:676, :]
# io.imsave("../image/vortex-gt-sub.png", gt_sub)
# io.imsave("../image/vortex-rnn-sub.png", rnn_sub)
# io.imsave("../image/vortex-lerp-sub.png", lerp_sub)


ssim_rnn = compare_ssim(gt, rnn, data_range=255., multichannel=True)
ssim_lerp = compare_ssim(gt, lerp, data_range=255., multichannel=True)
ssim_tsr_tvd = compare_ssim(gt, tsr_tvd, data_range=255., multichannel=True)
print("SSIM_LERP: {}".format(ssim_lerp))
print("SSIM_RNN: {}".format(ssim_rnn))
print("SSIM_TSR_TVD: {}".format(ssim_tsr_tvd))

