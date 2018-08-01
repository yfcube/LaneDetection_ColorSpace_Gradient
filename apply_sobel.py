import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobel = np.zeros_like(gray)
    if orient=='x':
        sobel = cv2.Sobel(gray,cv2.CV_64F, 1,0)
    elif orient=='y':
        sobel = cv2.Sobel(gray,cv2.CV_64F, 0,1)
    else:
        print("Orientation error")
        return sobel 
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_bin = np.zeros_like(scaled_sobel)
    sobel_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sobel_bin

if __name__ == '__main__':
    # Read in the image
    img = cv2.imread('signs_vehicles_xygrad.png',1)

    # Run the function
    grad_binary = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(grad_binary, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()