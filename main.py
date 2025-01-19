import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import segmentation as s
import cv2

plt.figure()

img = cv2.imread("Photos/image12.jpg")

gray=img.mean(axis=2)/255

v_edges=s.vertical_sobel(gray)

v_proj=s.vertical_projection(v_edges)

v_proj_blured=s.blur_vector(v_proj,np.ones(9)/9)

v_bands=s.band_detection(v_proj_blured.copy())

candidates=[]
for v_band in v_bands:
    blur_size = img.shape[1] // 5
    blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size
    h_edges=s.horizontal_sobel(gray[v_band[0]:v_band[1],:])
    h_proj=s.horizontal_projection(h_edges)
    h_proj_blured=s.blur_vector(h_proj,np.ones(blur_size)/blur_size)
    h_bands=s.band_detection(h_proj_blured.copy())
    for h_band in h_bands:
        h_band_proj=h_proj_blured[h_band[0]:h_band[1]]
        deriv=s.derivative(h_band_proj)
        h_range=s.analyze_derivative(deriv)
        projection=v_proj[v_band[0]:v_band[1]]
        candidates.append([v_band,[h_band[0]+h_range[0],h_band[0]+h_range[1]],s.heuristic(img,gray,[h_band[0]+h_range[0],h_band[0]+h_range[1]],v_band)])

sorted_candidates = sorted(candidates, key=lambda x: x[2])

matches = []
max = 0
for c in sorted_candidates:
    pom_l=c[1][0]-15
    if(pom_l<0):
        pom_l=0
    pom_r=c[1][1]+15
    if(pom_r>gray.shape[1]):
        pom_r=gray.shape[1]
    # plt.imshow(gray[c[0][0]:c[0][1],pom_l:pom_r],cmap='gray')
    # plt.show()
    image,thresh=s.preprocess_image(img[c[0][0]:c[0][1],pom_l:pom_r])
    contours=s.find_contours(thresh)
    sorted_characters=s.extract_characters(thresh,contours)
    if max < len(sorted_characters):
        max = len(sorted_characters)
        matches.clear()
        matches = sorted_characters
    # binary=s.otsu_binarization(gray[c[0][0]:c[0][1],pom_l:pom_r]*255)
    # bin_proj=s.horizontal_projection(binary)
    # char_separators=s.divide_into_chars(bin_proj.copy())
    # sorted_char_separators=sorted(char_separators)
    # print(sorted_char_separators)
    # #print(str(c)+"\n")
    # if(s.valid_candidate(sorted_char_separators,len(bin_proj))):
    #     n=len(sorted_char_separators)
    #     x_l=0
    #     plt.subplot(2,n//2+1,1)
    #     plt.imshow(binary,cmap='gray')
    #     for i in range(n):
    #         labeled_img, max_label=s.labelize_char_area(binary[:,x_l:sorted_char_separators[i]])
    #         print(str(max_label))
    #         char=s.analyze_labels(labeled_img,max_label)
    #         plt.subplot(2,n//2+2,i+2)
    #         plt.imshow(char,cmap='gray')
    #         x_l=sorted_char_separators[i]
    #     labeled_img, max_label=s.labelize_char_area(binary[:,x_l:])
    #     char=s.analyze_labels(labeled_img,max_label)
    #     plt.subplot(2,n//2+2,n+3)
    #     plt.imshow(char,cmap='gray')
    #     plt.show()
    # else:
    #     print("Candidate skipped")


s.save_and_display_characters(matches)