import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    sobel_size=15
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobel = np.zeros_like(gray)
    if orient=='x':
        sobel = cv2.Sobel(gray,cv2.CV_64F, 1,0, ksize=sobel_size)
    elif orient=='y':
        sobel = cv2.Sobel(gray,cv2.CV_64F, 0,1, ksize=sobel_size)
    else:
        print("Orientation error")
        return sobel 
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_bin = np.zeros_like(scaled_sobel)
    sobel_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sobel_bin

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F, 0,1, ksize=sobel_kernel)

    grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    result = np.zeros_like(gray)
    result[(grad_dir>=thresh[0])&(grad_dir<=thresh[1])]=1
    binary_output = np.copy(img) # Remove this line
    return result

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)): 
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F, 0,1, ksize=sobel_kernel)

    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mag_255 = np.uint8(255*grad_mag/np.max(grad_mag))

    grad_mag_bin = np.zeros_like(grad_mag_255)
    grad_mag_bin[(grad_mag_255>=mag_thresh[0])&(grad_mag_255<=mag_thresh[1])] = 1
    return grad_mag_bin

if __name__ == '__main__':
    # Read in an image
    image = mpimg.imread('signs_vehicles_xygrad.png')

    gradx = abs_sobel_thresh(image,'x',30,255)
    grady = abs_sobel_thresh(image,'y',30,255)
    mag_binary = mag_thresh(image,15,(50,255))
    dir_binary = dir_threshold(image,15,(-np.pi/8,np.pi/8))

    combined = np.zeros_like(dir_binary)
    combined1 = combined
    combined[((gradx == 1) & (grady == 1))] = 1
    combined1[(dir_binary==1)&(mag_binary==1)]=1
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(combined,cmap='gray')
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(combined1, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()