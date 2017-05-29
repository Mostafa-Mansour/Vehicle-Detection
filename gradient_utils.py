"""
This module contains functions that can be used to get Histogram of Oriented Gradient (HOG).
It contains the following functions:
		- get_hog_features 			:get Histogram of Oriented Gradient (HOG) of the features
		- extract_hog_features	:To get a raw features vector of the intensities and histogram

"""
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from color_utils import color_convert

from globals_variables import *

#----------------------------------------------------------------#
	
"""
# 1- get_hog_features

This function is used to get Histogram of Oriented Gradient (HOG) of the features

"""
# Define a function to return HOG features and visualization
def get_hog_features(img, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

#-------------------------------------------------------------------------------------------------#



"""
# 2- extract_hog_features

This function is used to:
	1- convert a list of images to a specific color space
	2- extract the HOG features of each image
	3- return a vector of HOG Features of each image
	
	input: list of images
	output: list of hog features of every image
"""
# Define a function to extract features from a list of images
def extract_hog_features(imgs, cspace=color_space, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image=color_convert(img=image,color_space=cspace)
        
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

#------------------------------------------------------------------------------------------#
if __name__=='__main__':
    import cv2
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 16
    cell_per_block = 4
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    img_car_rgb=mpimg.imread('./test_images/25.png')
    img_car=cv2.cvtColor(img_car_rgb,cv2.COLOR_RGB2GRAY)
    img_not_car_rgb=mpimg.imread('./test_images/extra40.png')
    img_not_car=cv2.cvtColor(img_not_car_rgb,cv2.COLOR_RGB2GRAY)
    print ("GETTING HOG")
    features_car, hog_car=get_hog_features(img=img_car,vis=True)
    features_car=np.array(features_car).ravel()
    features_not_car, hog_not_car=get_hog_features(img_not_car,vis=True)
    features_not_car=np.array(features_not_car).ravel()
    f,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(20,10))
    ax1.imshow(img_car_rgb)
    ax1.set_title("Car")
    ax2.imshow(hog_car,cmap='gray')
    ax2.set_title("HOG of a Car")
    ax3.imshow(img_not_car_rgb)
    ax3.set_title("Car")
    ax4.imshow(hog_not_car,cmap='gray')
    ax4.set_title("HOG of not a Car")
    #plt.imshow(features[0])
    plt.show()


	
 #    #exit()
    