import os
import glob
import random
import cv2 as cv
import shutil
import PIL.Image


random.seed(0)

root_dir = 'data/Siaya/Image'
sampled_dir = 'data/Siaya/ToSupervisely'

N = 200  # 120 images are eventually annotated

files = list(glob.glob(os.path.join(root_dir, '*.png')))

samples = random.sample(files, N)

for i, sample in enumerate(samples):
    folder = f'Batch{i % 5:02d}'
    os.makedirs(os.path.join(sampled_dir, folder), exist_ok=True)
    shutil.copyfile(
        sample,
        os.path.join(
            sampled_dir, folder, os.path.basename(sample)))
