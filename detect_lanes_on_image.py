import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('test_image.jpg')

# To make out the pixel values of the image
plt.imshow(image)
# plt.show()

cv2.imshow('Image', image)
cv2.waitKey(0)

def cvt_canny(image):
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply GuassianBlur, to reduce noise, with 5 X 5 kernal, with deviation = 0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('Blured Grayscale Image', blur)
    # cv2.waitKey(0)

    # Apply Canny function to convert high gradient change to white lines
    canny = cv2.Canny(blur, 50, 150)
    # cv2.imshow('Canny Image', canny)
    # cv2.waitKey(0)

    return canny

def region_of_interest(image):
    height = image.shape[0]
    # Triangle to mask
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    # Fill the mask with polygon, with white color
    cv2.fillPoly(mask, polygons, 255)
    # Bitwise and between mask and canny_image to get only roi with canny
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image) # Black image
    # Display lines on black image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=8)
    return line_image

def avg_slope_intercept(image, lines):
    left_fit = [] # coordinates on left
    right_fit = [] # coordinates on right

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Fit the points to a polynomial function
        parameters = np.polyfit((x1, x2), (y1, y2), deg=1) # Returns list of slope and y-intercept
        slope = parameters[0]
        intercept = parameters[1]

        # Left line has negative slope, right lines has positive slope
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Average all the slope and intercept to only get 2 lines for left and right
    left_fit_avg = np.average(left_fit, axis=0) # axis=0: vertically
    right_fit_avg = np.average(right_fit, axis=0)

    # Convert slope and intercept into x1, y1, x2, y2 to draw line
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)

    return np.array([left_line, right_line])


def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.01, 0 # Minimize the error if line_parameters could not unpack
    y1 = image.shape[0]
    y2 = int(y1*(2/5)) # lines will go from bottom to 3/5th of the image
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope) # From standard formula y = mx+b
    return np.array([x1, y1, x2, y2])

lane_image = np.copy(image)
canny_image = cvt_canny(lane_image)
cv2.imshow('Canny Image', canny_image)
cv2.waitKey(0)

roi_image = region_of_interest(canny_image)
cv2.imshow('Cropped ROI Image', roi_image)
cv2.waitKey(0)

# Finding lines
# image
# 2 - Pixels
# 1 - Precision (in radian), 1 degree ie pi/180
# 100 - threshold, min number of intersection in Hough space for a bin
# np.array([]) - placeholder array
# minLineLength - length of the line at accept as output
# maxLineGap - max distance between lines to consider those as a single line
lines = cv2.HoughLinesP(roi_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(image, lines)
cv2.imshow('Line Image', line_image)
cv2.waitKey(0)

# Combine image with only lines with actual image
combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, gamma=1)
cv2.imshow('Combined Image', combined_image)
cv2.waitKey(0)

# Optimizing the lines
avg_lines = avg_slope_intercept(lane_image, lines)
avg_line_image = display_lines(image, avg_lines)
cv2.imshow('Average Line Image', avg_line_image)
cv2.waitKey(0)

combined_avg_line_image = cv2.addWeighted(lane_image, 0.6, avg_line_image, 1, gamma=1)
cv2.imshow('Combined Average Line Image', combined_avg_line_image)
cv2.waitKey(0)
