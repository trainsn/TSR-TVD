from skimage import io
from skimage.measure import compare_ssim

gt = io.imread("../image/117-gt.png")
# print(gt)
# print(gt.shape)
# io.imshow(gt)

rnn = io.imread("../image/117-rnn.png")
lerp = io.imread("../image/117-lerp.png")

ssim_rnn = compare_ssim(gt, rnn, data_range=255., multichannel=True)
ssim_lerp = compare_ssim(gt, lerp, data_range=255., multichannel=True)
print("SSIM_RNN: {}".format(ssim_rnn))
print("SSIM_LERP: {}".format(ssim_lerp))

