import cv2
import glob
import numpy as np
import os
from tqdm import tqdm

input_dir='TrainLetters/'
output_dir='LettersImages/'

new_width = 50
new_height = 100


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each subdirectory in the input directory + progress bar by wrapping iterable into tqdm
    for subdir in tqdm(os.listdir(input_dir), desc="Processing directories"):
        sub_input_path = os.path.join(input_dir, subdir)
        sub_output_path = os.path.join(output_dir, subdir)

        if not os.path.exists(sub_output_path):
            os.makedirs(sub_output_path)

        for file_path in glob.glob(os.path.join(sub_input_path, "*.jpg")):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (new_width, new_height))
            #blurred= cv2.GaussianBlur(resized_img, (5, 5), 0)
            #thresh = cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #binary_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            output_file_path = os.path.join(sub_output_path, os.path.basename(file_path))

            cv2.imwrite(output_file_path, resized_img)

    print("All images have been processed and saved in:", output_dir)

process_images(input_dir, output_dir)