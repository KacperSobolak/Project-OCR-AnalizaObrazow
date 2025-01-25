import cv2
import glob
import numpy as np
import os
from tqdm import tqdm
from segmentation import otsu_binarization

input_dir='Characters'
output_dir='LettersImages/'

new_width = 50
new_height = 100


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in tqdm(os.listdir(input_dir), desc="Processing directories"):
        sub_input_path = os.path.join(input_dir, subdir)
        sub_output_path = os.path.join(output_dir, subdir)

        if not os.path.exists(sub_output_path):
            os.makedirs(sub_output_path)

        for file_path in glob.glob(os.path.join(sub_input_path, "*.jpg")):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (new_width, new_height))
            binary_img = otsu_binarization(resized_img)
            
            output_file_path = os.path.join(sub_output_path, os.path.basename(file_path))

            cv2.imwrite(output_file_path, binary_img)

    print("All images have been processed and saved in:", output_dir)

process_images(input_dir, output_dir)