import features_utils as f_utils
import windows_utils as w_utils
import classifier_utils as cl_utils 
import glob
import numpy as np
import pickle
from scipy.ndimage.measurements import label
from multi_detections_utils import *
import numpy as np
#from globals import *

def main(image_src):
    image=image_src.astype(np.float32)/255
    # # 1-loading data set
    # #cars path
    # cars_path="./KITTI_extracted/*.png"
    # #non cars path
    # notcars_path="./Extras/*.png"
    # #list of cars
    # cars=glob.glob(cars_path)
    # #list of non cars
    # not_cars=glob.glob(notcars_path)


    # #2- extracting features of cars and not-cars images
    # car_features=f_utils.extract_features(cars,color_space='YCbCr',bins_range=(0, 1),hog_channel='ALL')
    # notcar_features=f_utils.extract_features(not_cars,color_space='YCbCr',bins_range=(0, 1),hog_channel='ALL')

    # #3- convert cars_features list and notcars_features list to a vector of features
    # # to be used for training
    # X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # print(X.shape)                       

    # #4- Scale the features vector using "scale_vector" from "classifier_utils"
    # X_scaler=cl_utils.scale_vector(X)
    
    
    # scaled_X=X_scaler.transform(X)
    # #print(scaled_X.shape)

    # #5- Get the labels vector (1 for cars and 0 for notcars)
    # y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # #print(y.shape)

    # #6- Split the data set to training set and test set and shuffle their contents
    # # for that, "get_train_test_set" from "classifier_utils" is used
    # x_train,x_test,y_train,y_test=cl_utils.get_train_test_set(scaled_X=scaled_X,y=y,test_size=0.2)
    # # print(x_train.shape)
    # # print(y_train.shape)
    # # print(x_test.shape)
    # # print(y_test.shape)

    # #7- Training a classifier using "classifier_fit" from "classifier_utils"
    # svc=cl_utils.classifier_fit(x_train,y_train,classifier='linear_svm')

    # #8- check the accuracy of a classifier using "classifier_accuracy" from "classifier_utils"
    # #print("The accuracy of the classifier is {}".format(cl_utils.classifier_accuracy(x_test,y_test,svc)))

    X_scaler=pickle.load(open("X_scaler.p","rb"))
    svc=pickle.load(open("svc.p","rb"))
    #9- make a copy of the input image to be used later
    img_copy=np.copy(image)

    #10- make a slide window over the image
    
    # hyper parameters 
     #check the bottom half only of the image
    y_start_stop=[400,650] 
    #size of each window
    xy_window=(64, 64)
    #overlap between windows
    xy_overlap=(0.8, 0.8)


    # "slide_windows" from "windows_utils" will be used to give a list of windows over
    # the space determined before using x_start_stop and y_start_stop  
    #windows = w_utils.slide_windows(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
     #               xy_window=xy_window, xy_overlap=xy_overlap)

    #print(windows)
    
    #11- in each of the previous windows, we will check if this window contains a car or not car
    #to do that "search_windows" from "windows_utils" is used. the output of this function is a list of
    #of windows where cars were predicted
    #hot_windows=w_utils.search_windows(img=image,windows=windows,clf=svc,scaler=X_scaler,hog_channel='ALL')
    #print(hot_windows)

    #12- draw boxes around the hot windows using "draw_boxes" from "windows_utils"
    #window_img = w_utils.draw_boxes(img_copy, hot_windows, color=(0, 0, 255), thick=6)

    ##Parameters
    ### TODO: Tweak these parameters and see how the results change.
    # color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # orient = 9  # HOG orientations
    # pix_per_cell = 8 # HOG pixels per cell
    # cell_per_block = 2 # HOG cells per block
    # hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    # spatial_size = (32, 32) # Spatial binning dimensions
    # hist_bins = 32    # Number of histogram bins
    # spatial_feat = True # Spatial features on or off
    # hist_feat = True # Histogram features on or off
    # hog_feat = True # HOG features on or off
    
    #13- using "find_cars" from "features_utils"
    #X_scaler=StandardScaler.fit()
    img_find_car,heat_map=f_utils.find_cars(img=image, ystart=400, ystop=650, scale=1.5, svc=svc,X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)                    
    # f,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
    # ax1.imshow(img_find_car)
    # ax1.set_title('Multi detections')
    # ax2.imshow(heat_map,cmap='hot')
    # ax2.set_title('Heat map')
    # plt.show()
    # exit()
    heat_map=apply_threshold(heat_map,1)
    labels=label(heat_map)
    draw_img=draw_labeled_bboxes(image_src,labels)
    #draw_img=np.int(draw_img*255)

    return draw_img #, heat_map














if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from globals_variables import *
    from multi_detections_utils import *
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    img=mpimg.imread('./test_images/test6.jpg')
    # img=img.astype(np.float32)/255
    draw_img=main(img)
    plt.imshow(draw_img)
    plt.show()
    #exit()
    video_output = 'video.mp4'
    clip1 = VideoFileClip('./project_video.mp4')
    video_clip = clip1.fl_image(main) #NOTE: this function expects color images!!
    video_clip.write_videofile(video_output, audio=False)
    #print(pix_per_cell)
    #exit()
    #from sklearn.preprocessing import StandardScaler
    # img=mpimg.imread('./test_images/test6.jpg')
    # #convert from jpg to png
    # img=img.astype(np.float32)/255
    # img1,img2=main(img)
    # f,(ax1,ax2)=plt.subplots(1,2,figsize=(10,20))
    # ax1.imshow(img1)
    # ax2.imshow(img2,cmap='hot')
    # plt.show()