import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model

def segment_characters(license_plate_image):
    segmentation_spacing = 0.85

    img_gray = cv2.cvtColor(license_plate_image, cv2.COLOR_RGB2GRAY)
    ret, img_threshold = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    white = []
    black = []
    height, width = img_threshold.shape

    white_max = 0
    black_max = 0

    for i in range(width):
        white_count = 0
        black_count = 0
        for j in range(height):
            if img_threshold[j, i] == 255:
                white_count += 1
            else:
                black_count += 1

        white.append(white_count)
        black.append(black_count)

    white_max = max(white)
    black_max = max(black)

    def find_end(start):
        end = start + 1
        for m in range(start + 1, width - 1):
            if black[m] > segmentation_spacing * black_max:
                end = m
                break
        return end

    n = 1
    start = 1
    end = 2
    characters = []
    
    while n < width - 1:
        n += 1
        if white[n] > (1 - segmentation_spacing) * white_max:
            start = n
            end = find_end(start)
            n = end
            if end - start > 5: 
                character = img_threshold[1:height, start:end]
                characters.append(character)
                cv2.imshow('character', character)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    return characters


def recognize_characters(model, char_images, characters): # placeholder na funkcje
    return "brak" 



if __name__ == "__main__":
    # model = load_model("character_recognition_model.keras")

    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    license_plate_path = sys.argv[1]  
    license_plate_image = cv2.imread(license_plate_path)

    if license_plate_image is None:
        print("Nie można wczytać obrazu tablicy rejestracyjnej.")
        exit()

    char_images = segment_characters(license_plate_image)
    if not char_images:
        print("Nie udało się zsegmentować żadnych znaków.")
        exit()

    # recognized_text = recognize_characters(model, char_images, characters)

    # print("Odczytana tablica rejestracyjna:", recognized_text)
