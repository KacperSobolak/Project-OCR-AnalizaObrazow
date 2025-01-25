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

# def horizontal_edge_detection(gray):
#     kernel=np.zeros((3,3))
#     kernel[0,:]=-1
#     kernel[2,:]=1
#     return abs(apply_kernel(gray,kernel)) 

def vertical_edge_detection(gray): 
    kernel=np.zeros((3,3))
    kernel[:,0]=-1
    kernel[:,2]=1
    return abs(apply_kernel(gray,kernel)) 

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

# def horizontal_blur(im):
#     return apply_kernel(im,np.ones((1,9))/(1*9))

# def vertical_blur(im):
#     return apply_kernel(im,np.ones((9,1))/(1*9))

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

def generate_gaussian_kernel(size,sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    ay = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussian_adaptive_threshold(img_plate, block_size=11, C=2, sigma=1.0):
    kernel = generate_gaussian_kernel(block_size, sigma)
    
    binary = apply_kernel(img_plate,kernel,'reflect')
    
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            local_threshold = binary[i,j] - C/255

            if img_plate[i, j] > local_threshold:
                binary[i, j] = 1
            else:
                binary[i, j] = 0
    
    return binary

def otsu_binarization(image):
    # Compute the histogram
    hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))
    
    # Total number of pixels
    total_pixels = image.size
    
    # Initialize variables
    current_max_variance = 0
    best_threshold = 0
    sum_total = np.dot(np.arange(256), hist)  # Sum of all pixel values
    sum_background = 0
    weight_background = 0
    weight_foreground = 0

    for threshold in range(256):
        # Update background weight and sum
        weight_background += hist[threshold]
        if weight_background == 0:
            continue

        # Update foreground weight
        weight_foreground = total_pixels - weight_background

        if weight_foreground == 0:
            break

        # Update background sum
        sum_background += threshold * hist[threshold]

        # Calculate means
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Calculate between-class variance
        variance_between = (
            weight_background * weight_foreground *
            (mean_background - mean_foreground) ** 2
        )

        # Check if this is the best threshold
        if variance_between > current_max_variance:
            current_max_variance = variance_between
            best_threshold = threshold

    # Apply the optimal threshold to binarize the image
    binary_image = (image > best_threshold).astype(np.uint8) * 255

    return binary_image

def global_binarization(gray, precision=5):
    width,height=gray.shape
    dt=1
    t=0.5
    while dt>precision/255:
        n_higher=0
        n_lower=0
        sum_lower=0
        sum_higher=0
        for x in range(width):
            for y in range(height):
                if(gray[x,y]<t):
                    n_lower+=1
                    sum_lower+=gray[x,y]
                else:
                    n_higher+=1
                    sum_higher+=gray[x,y]
        temp=t
        if(n_lower==0):
            avg_lower=0
        else:
            avg_lower=sum_lower/n_lower
        if(n_higher==0):
            avg_higher=0
        else:
            avg_higher=sum_higher/n_higher
        t=(avg_lower+avg_higher)/2
        dt=t-temp
    return (gray > t).astype(np.uint8)
    
def divide_into_chars(proj, cx=0.7, cw=0.85):
    v_max=np.max(proj)
    division_points=[]
    while True:
        x_max=np.argmax(proj)
        if(proj[x_max]==0):
            return division_points
        x=x_max-1
        while x>=0 and proj[x]>proj[x_max]*cx:
            x-=1
        if(x!=-1):
            x_l=x
            x=x_max+1
            while x<proj.size and proj[x]>proj[x_max]*cx:
                x+=1
            if(x!=proj.size):
                x_r=x
                if(proj[x_max]<cw*v_max):
                    return sorted(division_points)
                proj[x_l:x_r]=0
                division_points.append(x_max)
            else:
                proj[x_l:]=0
        else:
            pom=x_max+1
            while pom<proj.size and proj[pom]>proj[pom-1]:
                pom+=1
            proj[:pom+1]=0

def valid_candidate(char_separators, width,threshold=0.33):
    widths=[]
    n=len(char_separators)
    if(n<6):
        return False
    for i in range(1,n):
        widths.append(char_separators[i]-char_separators[i-1])

    mean=np.mean(widths)
    std=np.std(widths)
    # print("mean: "+str(mean))
    # print("std: "+str(std))
    # print(str(std-threshold*mean))
    if std<threshold*mean:
        return True
    return False

def flood_fill(image, seed_point, new_value):
    rows, cols = image.shape
    x, y = seed_point
    target_value = image[x, y]
    
    if target_value == new_value:
        return image
    
    stack = [(x, y)]
    
    while stack:
        cx, cy = stack.pop()
        if image[cx, cy] == target_value:
            image[cx, cy] = new_value
            
            if cx > 0:
                stack.append((cx - 1, cy))
            if cx < rows - 1:
                stack.append((cx + 1, cy))
            if cy > 0:
                stack.append((cx, cy - 1))
            if cy < cols - 1:
                stack.append((cx, cy + 1))

    return image

def labelize_char_area(img):
    rows, cols = img.shape
    cur_label=2
    for i in range(rows):
        for j in range(cols):
            if(img[i,j]==0):
                img=flood_fill(img,[i,j],cur_label)
                cur_label+=1
    return img, cur_label-1

def analyze_labels(img,max_label):
    best_candidate=[]
    best_heuristic=1
    height,width=img.shape
    for cur_label in range(2,max_label+1):
        x_min=width-1
        x_max=0
        y_min=height-1
        y_max=0
        count=0
        for y in range(height):
            for x in range(width):
                if(img[y,x]==cur_label):
                    count+=1
                    if(x<x_min):
                        x_min=x
                    if(x>x_max):
                        x_max=x
                    if(y<y_min):
                        y_min=y
                    if(y>y_max):
                        y_max=y
        heuristic=count/((x_max-x_min+1)*(y_max-y_min+1))
        if(heuristic<best_heuristic):
            best_candidate=[x_min,x_max,y_min,y_max, cur_label]
            best_heuristic=heuristic
    return img[best_candidate[2]:best_candidate[3]+1,best_candidate[0]:best_candidate[1]+1]==best_candidate[4]


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
    characters=[cv2.resize(char,(needed_width,needed_height)) for char in characters]   # black images on white background
    inverted_characters = [cv2.bitwise_not(char) for char in characters ]   # black images on white background
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

def save_and_display_characters(characters):
    for idx, char in enumerate(characters):
        char_resized = cv2.resize(char, (50, 100))
        plt.subplot(1, len(characters), idx + 1)
        plt.imshow(char_resized, cmap='gray')
        plt.axis('off')
    plt.show()