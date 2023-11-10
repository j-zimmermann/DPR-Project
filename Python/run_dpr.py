from skimage.io import imread
from tifffile import imwrite
import os
from dpr import DPR_UpdateSingle
import numpy as np
from tqdm import tqdm

data_folder = "../MatLab/Test_image"
save_folder = "DPR_image"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
# data_name = "test_image"
data_name = "sarcomere"
ending = ".tif"
file_name = data_name + ending


img = imread(os.path.join(data_folder, file_name))
num_frames = img.shape[0]  # frame number

PSF = 4
gain = 2
background = 10
temporal = "mean"


new_shape = round(5.0 * img.shape[1] / PSF * 1.6651)
I_DPR = np.zeros(shape=(img.shape[0], new_shape - 1, new_shape - 1))
raw_magnified = []
# process single
for i in tqdm(range(num_frames)):
    image_in = img[i]
    single_I_DPR, single_raw_mag = DPR_UpdateSingle(image_in, PSF, gain, background)
    I_DPR[i, :] = single_I_DPR
    raw_magnified.append(single_raw_mag)

raw_magnified = np.array(raw_magnified)
temporal_options = ["mean", "var"]
if temporal in temporal_options:
    if temporal == "mean":
        I_DPR = np.mean(I_DPR, axis=0)
        raw_magnified = np.mean(raw_magnified, axis=0)
    else:
        I_DPR = np.var(I_DPR, axis=0)
        raw_magnified = np.var(raw_magnified, axis=0)

imwrite(os.path.join(save_folder, data_name + "_DPR_image.tif"), I_DPR)
imwrite(os.path.join(save_folder, data_name + "_DPR_image_magnified.tif"), raw_magnified)
