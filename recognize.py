import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
                  'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
                  'L': 21, 'M': 22, 'N': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29, 'V': 30,
                  'W': 31, 'X': 32, 'Y': 33, 'Z': 34}

needed_height = 100
needed_width = 50

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


def get_text_from_character_images(characters):
    characters=[cv2.resize(char,(needed_width,needed_height)) for char in characters]   # black images on white background
    inverted_characters = [cv2.bitwise_not(char) for char in characters ]   # black images on white background
    reshaped_input_characters=[np.expand_dims(char, axis=-1) for char in inverted_characters]
    normalized_input_characters=[char/255.0 for char in reshaped_input_characters]


    our_cnn_model=tf.keras.models.load_model('models/char_recognition_cnn_v2.keras')
    print('Loaded model')
    
    text=""
    
    for char in normalized_input_characters:
        char_with_batch=np.expand_dims(char,axis=0)
        prediction=our_cnn_model.predict(char_with_batch)
        predicted_label=int(np.argmax(prediction))
        for key,value in dictionary.items():
            if value==predicted_label:
                text+=key
   
    return text


def main(image_path):
    image, thresh = preprocess_image(image_path)
    contours = find_contours(thresh)
    characters = extract_characters(thresh, contours)   # white images on black background
    
    
    text=get_text_from_character_images(characters)
    print("Odnaleziony tekst to: "+text)
    
    save_and_display_characters(characters)

    
main(sys.argv[1])