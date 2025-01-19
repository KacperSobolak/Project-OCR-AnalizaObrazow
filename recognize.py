import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding for better segmentation
    ret, img_threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    return image, img_threshold

def find_contours(thresh):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_characters(image, contours):
    characters = []
    char_dimensions = []

    # Get image dimensions
    image_height, image_width = image.shape[:2]

    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Define size thresholds based on image size
        min_h = image_height * 0.15  # Minimum height as a percentage of image height
        min_w = image_width * 0.06   # Minimum width as a percentage of image width
        max_h = image_height * 1  # Minimum height as a percentage of image height
        max_w = image_width * 0.4   # Minimum width as a percentage of image width

        # Filter out small regions that are unlikely to be characters
        if h > min_h and w > min_w and h < max_h and w < max_w:  # Adjust based on your license plate size
            char_roi = image[y:y+h, x:x+w]
            characters.append(char_roi)
            char_dimensions.append((x, y, w, h))

    # Sort characters by their x-coordinate for proper sequence
    sorted_indices = sorted(range(len(characters)), key=lambda i: char_dimensions[i][0])
    sorted_characters = [characters[i] for i in sorted_indices]

    return sorted_characters

def save_and_display_characters(characters):
    for idx, char in enumerate(characters):
        # Resize for visualization
        char_resized = cv2.resize(char, (50, 100))
        plt.subplot(1, len(characters), idx + 1)
        plt.imshow(char_resized, cmap='gray')
        plt.axis('off')
    plt.show()

def main(image_path):
    image, thresh = preprocess_image(image_path)
    contours = find_contours(thresh)
    characters = extract_characters(thresh, contours)
    save_and_display_characters(characters)

# Example usage
main(sys.argv[1])