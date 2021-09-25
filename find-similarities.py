import cv2
import os
from sys import argv
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

img = argv[1]

test_img = cv2.imread(img)

ssim_measures = {}
rmse_measures = {}
measures1 = {}
measures2 = {}
measures3 = {}
measures4 = {}
measures5 = {}
measures6 = {}
measures7 = {}
measures8 = {}
measures9 = {}

scale_percent = 10  # percent of original img size
width = int(test_img.shape[1] * scale_percent / 100)
height = int(test_img.shape[0] * scale_percent / 100)
dim = (width, height)

test_img = cv2.resize(test_img, dim, interpolation=cv2.INTER_AREA)
# test_img = cv2.fastNlMeansDenoisingColored(test_img,None,15,15,7,21)

data_dir = 'dataset'

for file in os.listdir(data_dir):
  if file.endswith('.jpg') and not file.endswith(img):
    img_path = os.path.join(data_dir, file)
    data_img = cv2.imread(img_path)
    resized_img = cv2.resize(data_img, dim, interpolation=cv2.INTER_AREA)
    # resized_img = cv2.fastNlMeansDenoisingColored(resized_img,None,15,15,7,21)
    ssim_measures[img_path] = ssim(test_img, resized_img)
    rmse_measures[img_path] = rmse(test_img, resized_img)
    measures1[img_path] = mse(test_img, resized_img)
    measures2[img_path] = msssim(test_img, resized_img)
    measures3[img_path] = ergas(test_img, resized_img)
    measures4[img_path] = psnr(test_img, resized_img)
    measures5[img_path] = uqi(test_img, resized_img)
    measures6[img_path] = scc(test_img, resized_img)
    measures7[img_path] = rase(test_img, resized_img)
    measures8[img_path] = sam(test_img, resized_img)
    measures9[img_path] = vifp(test_img, resized_img)


def calc_closest_val(dict, checkMax):
  result = {}
  if (checkMax):
    closest = max(dict.values())
  else:
    closest = min(dict.values())

  for key, value in dict.items():
    # print("The difference between ", key, " and the original image is : \n", value)
    if (value == closest):
      result[key] = closest

  # print("The closest value: ", closest)
  # print("######################################################################")
  return result


use_max = True

ssim = calc_closest_val(ssim_measures, use_max)
rmse = calc_closest_val(rmse_measures, use_max)
ret_mse = calc_closest_val(measures1, use_max)
ret_msssim = calc_closest_val(measures2, use_max)
ret_ergas = calc_closest_val(measures3, use_max)
ret_psnr = calc_closest_val(measures4, use_max)
ret_uqi = calc_closest_val(measures5, use_max)
ret_scc = calc_closest_val(measures6, use_max)
ret_rase = calc_closest_val(measures7, use_max)
ret_sam = calc_closest_val(measures8, use_max)
ret_vifp = calc_closest_val(measures9, use_max)

print("The most similar according to SSIM: ", ssim)
print("The most similar according to RMSE: ", rmse)
print("The most similar according to MSE: ", ret_mse)
print("The most similar according to msssim: ", ret_msssim)
print("The most similar according to ergas: ", ret_ergas)
print("The most similar according to psnr: ", ret_psnr)
print("The most similar according to uqi: ", ret_uqi)
print("The most similar according to rase: ", ret_rase)
print("The most similar according to sam: ", ret_sam)
print("The most similar according to ergas: ", ret_vifp)
