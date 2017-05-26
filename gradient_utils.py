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

#----------------------------------------------------------------#
	
"""
# 1- get_hog_features

This function is used to get Histogram of Oriented Gradient (HOG) of the features

"""
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
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
def extract_hog_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
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
	colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9
	pix_per_cell = 16
	cell_per_block = 4
	hog_channel = 0 # Can be 0, 1, 2, or "ALL"
	img=mpimg.imread('25.png')
	features=extract_hog_features(['25.png'])
	features=np.array(features).ravel()
	plt.plot(features)
	#plt.imshow(features[0])
	plt.show()
