"""
This module contains functions that can be used to get different windows on an image.
It contains the following functions:
		- draw_boxes 				:To draw boxes at images
		- slide_windows				:To get slide windows on an image

"""
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------#

# 1- draw_boxes
"""
This function is used to draw rectangular boxes on a certain image

Input: 	- img:image
		- bboxes:list of boxes
		- color:color of the boxes
		- thick: thick of the boxes

output:	copy of the input img with drawn boxes

"""
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

#--------------------------------------------------------------------------------------------#

# slide_windows
"""
This function is used to get  slide windows in an image
inputs:	- img:image
		- x_start_stop:starting and stopping x coordinate of the searching area
		- y_start_stop:starting and stopping y coordinate of the searching area
		- xy_window: size of the window
		- xy_ovelap: overlapping between windows

output: a list of windows that can be drawn later
"""

def slide_windows(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0]==None:
        x_start_stop[0]=0
    if x_start_stop[1]==None:
        x_start_stop[1]=img.shape[1]
    if y_start_stop[0]==None:
        y_start_stop[0]=0
    if y_start_stop[1]==None:
        y_start_stop[1]=img.shape[0]
    # Compute the span of the region to be searched
    xspan=x_start_stop[1]-x_start_stop[0]
    yspan=y_start_stop[1]-y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range (ny_windows):
        for xs in range (nx_windows):
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx,starty),(endx,endy)))
    # Return the list of windows
    return window_list

#--------------------------------------------------------------------------------------------#
if __name__ == '__main__':
	img=mpimg.imread('test6.jpg')
	windows=slide_windows(img,y_start_stop=[np.int(img.shape[0]/2),img.shape[0]])
	imgcopy=draw_boxes(img,windows)
	plt.imshow(imgcopy)
	plt.show() 
