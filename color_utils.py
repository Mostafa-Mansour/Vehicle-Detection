"""
This module contains functions that can be used to get color features.
It contains the following functions:
		- color_convert 		:convert images to different color spaces
		- bin_spatial			:get a raw features vector of pixel intensities 
		- color_hostogram 		:To get the histogram of each image channel
		- extract_color_features:To get a raw features vector of the intensities and histogram

"""
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#----------------------------------------------------------------#

"""
# 1- color_convert

color_convert : to convert an image to different current space
inputs: img: src image in RGB
		color_space: wanted color_space

output: img: dst image
"""  
def color_convert(img,color_space='RGB'):
	
	if color_space != 'RGB':
		if color_space=='HSV':
			img_copy=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
		elif color_space=='HLS':
			img_copy=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
		elif color_space=='GRAY':
			img_copy=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		elif color_space=='BGR':
			img_copy=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
		elif color_space=='YCrCb':
			img_copy=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
		elif color_space=='LUV':
			img_copy=cv2.cvtColor(img,cv2.COLOR_RGB2LUV)
		elif color_space=='YUV':
			img_copy=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
	else:
		img_copy=np.copy(img)

	return img_copy


#-----------------------------------------------------------------#

"""
# 2- bin_spatial

bin_spatial : This function is used to  resize and image and get a vector of raw intensities

"""
def bin_spatial(img,size=(32,32)):
	features=cv2.resize(img,size).ravel()
	return features

#------------------------------------------------------------------#

"""
# 3- color_hostogram

This function is used to get the histogram of every channel of an image
"""
def color_histogram(img,nbins=32,bins_range=(0,256)):
	
	# Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#---------------------------------------------------------------------------------------------#


"""
# 4- extract_color_features

This function is used to:
	1- convert an image to a desirable color space
	2- extract the raw intensity features from the image using bin_spatial function
	3- extract the histogram of each image channel using color_histogram function
	4- extract hog_features of of the image
	5- concatenate the raw features with histogram in one feature vector

	input : a list of images
	output: raw and histogram features
"""

def extract_color_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        
        # apply color conversion if other than 'RGB'
        feature_image=color_convert(image,color_space=cspace)
             
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #plt.plot(spatial_features)
        #plt.show()
        # Apply color_hist() also with a color space option now
        hist_features = color_histogram(feature_image, nbins=hist_bins, bins_range=hist_range)
        #plt.plot(hist_features)
        #plt.show()
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
        #plt.plot(features)
        #plt.show()
    # Return list of feature vectors
    return features

#-------------------------------------------------------------------------------------#

if __name__ == '__main__':
	img=mpimg.imread('25.png')
	raw_features=bin_spatial(img)
	histogram_features=color_histogram(img,bins_range=(0,1))
	features=extract_color_features(['25.png'],hist_range=(0,1))
	features=np.array(features).ravel()
	f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(20,10))
	ax1.plot(raw_features)
	ax1.set_title('raw intensity features')
	ax2.plot(histogram_features)
	ax2.set_title('histogram feature')
	ax3.plot(features)
	ax3.set_title('raw intensity and histogram features')
	plt.show()