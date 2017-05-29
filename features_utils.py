"""
This module contains a function that used to get raw intensity features, histogram features ans HOG features
"""

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from classifier_utils import scale_vector

from color_utils import color_convert, bin_spatial, color_histogram
from gradient_utils import get_hog_features

from multi_detections_utils import add_heat
from globals_variables import * 

from multi_detections_utils import *
from scipy.ndimage.measurements import label


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, bins_range=(0,1), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image=color_convert(img,color_space=color_space)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_histogram(feature_image, nbins=hist_bins,bins_range=bins_range)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
#--------------------------------------------------------------------------------------------#
"""
# 2- extract_features 
This function do the same thing as single_img_features, except that it takes a list of images as an input and 
return the features of each image.
"""
def extract_features(imgs, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, bins_range=bins_range, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image=color_convert(image,color_space=color_space)
         

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            #print(len(spatial_features))
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_histogram(feature_image, nbins=hist_bins,bins_range=bins_range)
            #print(len(hist_features))
            file_features.append(hist_features)
        if hog_feat == True:
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
            #print(len(hog_features))
            file_features.append(hog_features)
        #print(len(file_features[0]))
        features.append(np.concatenate(file_features))
    # Return list of feature ix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new vectors
    return features

#------------------------------------------------------------------------------------------#
"""
3- find_car
This function can be used to extract features using hog sub-sampling and make predictions 
"""
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc,X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    
    
    heatmap=np.zeros_like(img[:,:,0])

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = color_convert(img_tosearch, color_space=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    img_boxes=[]
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (128,128))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_histogram(subimg, nbins=hist_bins,bins_range=(0,1))
            ##print(np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1).shape)
            #exit()
            # print(hog_features.shape)
            # exit()
            # all_features=[]
            # all_features.append(spatial_features)
            # all_features.append(hist_features)
            # all_features.append(hog_features)
            # all_features=np.concatenate(all_features)
            # Scale features and make a prediction
            #print(spatial_features.shape)
            #print(hist_features.shape)
            #print(hog_features.shape)
            #X_scaler=scale_vector(np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1))
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1))    
            #print(test_features.shape)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            #exit()
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                
                # print(ytop_draw+ystart)
                # print(ystart.type)
                # print(ytop_draw+win_draw+ystart)
                # print(xbox_left)
                # print(xbox_left+win_draw)
                # #exit()
                img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)) )
                # img_boxes.append((int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart)))
                heatmap=add_heat(heatmap,[((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))])
                # print(heatmap) 
                
    return draw_img, heatmap
if __name__ == '__main__':
	import pickle
	X_scaler=pickle.load(open("X_scaler.p","rb"))
	features=extract_features(['./test_images/25.png'])
	f,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
	ax1.plot(np.array(features).ravel())
	ax1.set_title("Unnormalized features vector ")
	ax2.plot(X_scaler.transform(np.array(features).ravel()))
	ax2.set_title("Normalized features vector")
	plt.show()
    # img=mpimg.imread('./test_images/test6.jpg')
    # img=img.astype(np.float32)/255
    # import pickle
    # #features=single_img_features(img)
    # #features_=extract_features(['./test_images/25.png'])
    # #print(features_.shape)
    # svc=pickle.load(open("svc.p","rb"))
    # X_scaler=pickle.load(open("X_scaler.p","rb"))
    # draw_img, heatmap=find_cars(img=img, ystart=400, ystop=650, scale=0.75, svc=svc,X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)
    # heatmap=apply_threshold(heatmap,1)
    # labels=label(heatmap)
    # draw_img=draw_labeled_bboxes(img,labels)
    # f, (ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
    # ax1.imshow(draw_img)
    # ax2.imshow(heatmap,cmap='hot')
    # plt.show()