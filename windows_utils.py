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


"""
# 1- draw_boxes

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

"""
# 2- slide_windows

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

"""
# 3- search_window
This function is used to get the feature of every slide window in an image
"""
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #sklearn>=0.18
from sklearn.svm import LinearSVC
from features_utils import single_img_features
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier

        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
#-------------------------------------------------------------------------------------------#

if __name__ == '__main__':
	img=mpimg.imread('./test_images/test6.jpg')
	windows=slide_windows(img,y_start_stop=[np.int(img.shape[0]/2),img.shape[0]])
	imgcopy=draw_boxes(img,windows)
	plt.imshow(imgcopy)
	plt.show() 
