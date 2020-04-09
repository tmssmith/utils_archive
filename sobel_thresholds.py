import cv2
import numpy as np

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

def sobel_abs_thresh(img, orient='x', sobel_kernel = 3, thresh = (0,255)):
    gray = grayscale(img)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    else:
        print("Orientation must be 'x' or 'y'")
        return
    abs_sobel = np.absolute(sobel)
    abs_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mask = cv2.inRange(abs_sobel, thresh[0], thresh[1])

    return mask

def sobel_mag_thresh(img, sobel_kernel = '3', thresh = (0,255)):
    gray = grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    mag_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    mag_sobelxy = 255 * mag_sobelxy / np.max(mag_sobelxy)
    mask = cv2.inRange(mag_sobelxy, thresh[0], thresh[1])

    return mask

def sobel_dir_thresh(img, sobel_kernel = '3', thresh = (0, np.pi / 2)):
    gray = grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_sobel = np.arctan2(abs_sobely, abs_sobelx)
    mask = cv2.inRange(grad_sobel, thresh[0], thresh[1])
    
    return mask

k_size = 3
img = cv2.imread("Images/sobel_test.png")

gradx = sobel_abs_thresh(img, 'x', k_size, (20,100))
grady = sobel_abs_thresh(img, 'y', k_size, (20,100))
sobel_mag = sobel_mag_thresh(img, 9, (30,100))
sobel_grad = sobel_dir_thresh(img, 15, (0.7, 1.3))

combined = cv2.cvtColor(
    (gradx & grady) | (sobel_mag & sobel_grad),
    cv2.COLOR_GRAY2BGR
    )


overlay = cv2.addWeighted(img, 0.3, combined, 0.7, 0)

cv2.imshow('combined', overlay)
cv2.waitKey(0)
