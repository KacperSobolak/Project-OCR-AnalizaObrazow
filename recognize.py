import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 10)     
    

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return image, thresh  

def extract_characters(image, contours):
    characters = []
    char_dimensions = []

    image_height, image_width = image.shape[:2] 

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        min_h = image_height * 0.4 
        min_w = image_width * 0.04  
        max_h = image_height * 0.8   
        max_w = image_width * 0.3   

        if min_h < h < max_h and min_w < w < max_w:
            char_roi = image[y:y+h, x:x+w]
            characters.append(char_roi)
            char_dimensions.append((x, y, w, h))

    sorted_indices = sorted(range(len(characters)), key=lambda i: char_dimensions[i][0])
    sorted_characters = [characters[i] for i in sorted_indices]

    return sorted_characters


def find_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def save_and_display_characters(characters):
    plt.figure(figsize=(10, 5))
    for idx, char in enumerate(characters):
        char_resized = cv2.resize(char, (50, 100))
        plt.subplot(1, len(characters), idx + 1)
        plt.imshow(char_resized, cmap='gray')
        plt.axis('off')
    plt.show()


def main(image_path):
    image, thresh = preprocess_image(image_path)
    plt.figure()
    plt.imshow(thresh, cmap='gray')
    contours = find_contours(thresh)
    characters = extract_characters(thresh, contours)   # white images on black background
    
    
    inverted_characters = [cv2.bitwise_not(char) for char in characters ]   # black images on white background

    save_and_display_characters(characters)
    save_and_display_characters(inverted_characters)

main(sys.argv[1])