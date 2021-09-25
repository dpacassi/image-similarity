import cv2
import os
from sys import argv
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

img = argv[1]

test_img = cv2.imread(img)

ssim_measures = {}
rmse_measures = {}

scale_percent = 10  # percent of original img size
width = int(test_img.shape[1] * scale_percent / 100)
height = int(test_img.shape[0] * scale_percent / 100)
dim = (width, height)

test_img = cv2.resize(test_img, dim, interpolation=cv2.INTER_AREA)

data_dir = 'dataset'

for file in os.listdir(data_dir):
  if file.endswith('.jpg'):
    img_path = os.path.join(data_dir, file)
    data_img = cv2.imread(img_path)
    resized_img = cv2.resize(data_img, dim, interpolation=cv2.INTER_AREA)
    ssim_measures[img_path] = ssim(test_img, resized_img)
    rmse_measures[img_path] = rmse(test_img, resized_img)


def calc_closest_val(dict, checkMax):
  result = {}
  if (checkMax):
    closest = max(dict.values())
  else:
    closest = min(dict.values())

  for key, value in dict.items():
    print("The difference between ", key, " and the original image is : \n", value)
    if (value == closest):
      result[key] = closest

  print("The closest value: ", closest)
  print("######################################################################")
  return result


ssim = calc_closest_val(ssim_measures, True)
rmse = calc_closest_val(rmse_measures, False)

print("The most similar according to SSIM: ", ssim)
print("The most similar according to RMSE: ", rmse)
