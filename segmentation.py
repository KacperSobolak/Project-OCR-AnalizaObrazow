import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

needed_height = 100
needed_width = 50

dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
                  'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
                  'L': 21, 'M': 22, 'N': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29, 'V': 30,
                  'W': 31, 'X': 32, 'Y': 33, 'Z': 34}

def apply_kernel(im, kernel,pad_mode='edge'):
    im_rows, im_cols=im.shape
    k_rows, k_cols=kernel.shape

    if(k_rows%2!=1 or k_cols%2!=1):
        print("kernel should be of even size")
        return None
    if(im_rows<k_rows or im_cols<k_cols):
        print("Image is smaller than kernel (in some dimension)")
        return None
    
    result=np.zeros((im_rows,im_cols))
    n_row=(int)((k_rows-1)/2)
    n_col=(int)((k_cols-1)/2)
    
    pad_im=np.pad(im,((n_row,n_row),(n_col,n_col)),mode=pad_mode)

    for i in range(im_rows):
        for j in range(im_cols):
            matrix=pad_im[i:i+2*n_row+1,j:j+2*n_col+1]
            result[i,j]=np.sum(matrix*kernel)
    return result

def horizontal_sobel(gray):
    kernel=np.zeros((3,3))
    kernel[0,:]=-1
    kernel[0,1]=-2
    kernel[2,:]=1
    kernel[2,1]=2
    return abs(apply_kernel(gray,kernel)) 

def vertical_sobel(gray):
    kernel=np.zeros((3,3))
    kernel[:,0]=-1
    kernel[1,0]=-2
    kernel[:,2]=1
    kernel[1,2]=2
    return abs(apply_kernel(gray,kernel)) 

def sobel_edge_detection(gray):
    return horizontal_sobel(gray)+vertical_sobel(gray)

def horizontal_projection(edges_im):
    return np.sum(edges_im,axis=0)

def vertical_projection(edges_im):
    return np.sum(edges_im,axis=1)

def band_detection(proj,c=0.55,n=9):
    candidates=[]
    for i in range(n):
        y_max=np.argmax(proj)
        y=y_max-1
        while y>=0 and proj[y]>proj[y_max]*c:
            y-=1
        if(y!=-1):
            y0=y
            y=y_max+1
            while y<proj.size and proj[y]>proj[y_max]*c:
                y+=1
            if(y!=proj.size):
                y1=y
                candidates.append([y0,y1])
                proj[y0:y1+1]=0
            else:
                proj[y0:]=0
        else:
            pom=y_max+1
            while pom<proj.size and proj[pom]>proj[pom-1]:
                pom+=1
            proj[:pom]=0
    return candidates

def blur_vector(vec, kernel):
    vec_size=len(vec)
    f_size=len(kernel)

    if(f_size%2==0):
        print("kernel should be of even size")
        return None
    if(vec_size<f_size):
        print("vector is smaller than kernel")
        return None
    
    result=np.zeros(vec_size)
    n=(int)((f_size-1)/2)

    pad_vec=np.pad(vec,pad_width=n,mode='edge')

    for i in range(vec_size):
        part=pad_vec[i:i+f_size]
        result[i]=np.sum(part*kernel)
    
    return result

def derivative(proj,h=4):
    p_size=proj.size
    deriv=np.zeros(p_size)
    for i in range(h,p_size):
        deriv[i]=(proj[i]-proj[i-h])/h
    return deriv

def analyze_derivative(deriv,c=0.6):
    d_size=deriv.size
    der_max=np.max(deriv)
    der_min=np.min(deriv)
    xp0=0
    for i in range(1,(int)(np.floor(d_size/2))):
        if deriv[i]>=c*der_max:
            xp0=i
            break

    xp1=d_size-1
    for i in range(1,(int)(np.floor(d_size/2))):   
        if deriv[d_size-1-i]<=c*der_min:
            xp1=d_size-1-i
            break
    return [xp0,xp1]

def heuristic_size_ratio(h_band,v_band):
    return abs((h_band[1]-h_band[0])/(v_band[1]-v_band[0])-5)

def heuristic_rgb(original, gray,h_band,v_band):
    r=original[v_band[0]:v_band[1],h_band[0]:h_band[1],0]
    g=original[v_band[0]:v_band[1],h_band[0]:h_band[1],1]
    b=original[v_band[0]:v_band[1],h_band[0]:h_band[1],2]
    gray_area=gray[v_band[0]:v_band[1],h_band[0]:h_band[1]]
    r_error=np.mean(abs(r/255-gray_area))
    g_error=np.mean(abs(g/255-gray_area))
    b_error=np.mean(abs(b/255-gray_area))
    error=(r_error+g_error+b_error)/3
    return error*10

def heuristic_edge(gray,h_band,v_band):
    return 1/np.mean(sobel_edge_detection(gray[v_band[0]:v_band[1],h_band[0]:h_band[1]]))

def heuristic(rgb,gray,h_band,v_band):
    return heuristic_size_ratio(h_band,v_band)+heuristic_rgb(rgb,gray,h_band,v_band)+heuristic_edge(gray,h_band,v_band)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    return image, thresh

def extract_characters(image, contours):
    characters = []
    char_dimensions = []

    image_height, image_width = image.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        min_h = image_height * 0.5 
        min_w = image_width * 0.04  
        max_h = image_height * 1  
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

def get_text_from_character_images(characters):
    characters=[cv2.resize(char,(needed_width,needed_height)) for char in characters] 
    inverted_characters = [cv2.bitwise_not(char) for char in characters ]  
    reshaped_input_characters=[np.expand_dims(char, axis=-1) for char in inverted_characters]
    normalized_input_characters=[char/255.0 for char in reshaped_input_characters]


    our_cnn_model=tf.keras.models.load_model('models/char_recognition_cnn.keras')
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