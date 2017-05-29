import features_utils as f_utils
import windows_utils as w_utils
import classifier_utils as cl_utils 
import glob
import numpy as np
import pickle

from globals_variables import *

def classify():

	# 1-loading data set
    #cars path
    cars_path="./KITTI_extracted/*.png"
    #non cars path
    notcars_path="./Extras/*.png"
    #list of cars
    cars=glob.glob(cars_path)
    #print(len(cars))
    #list of non cars
    not_cars=glob.glob(notcars_path)
    #print(len(not_cars))
    #from other data bases
    cars_GTI_Right=glob.glob("/home/ros-indigo/Desktop/Vehicle detection/vehicles/GTI_Right/*.png")
    for item in cars_GTI_Right:
    	cars.append(item)
    cars_GTI_Left=glob.glob("/home/ros-indigo/Desktop/Vehicle detection/vehicles/GTI_Left/*.png")
    for item in cars_GTI_Left:
    	cars.append(item)
    cars_GTI_Far=glob.glob("/home/ros-indigo/Desktop/Vehicle detection/vehicles/GTI_Far/*.png")
    for item in cars_GTI_Far:
    	cars.append(item)
    not_cars_GTI=glob.glob("/home/ros-indigo/Desktop/Vehicle detection/non-vehicles/non-vehicles/GTI/*.png")
    for item in not_cars_GTI:
    	not_cars.append(item)
    #print(len(cars))
    #print(len(not_cars))
    #exit()

    #2- extracting features of cars and not-cars images
    car_features=f_utils.extract_features(cars,color_space=color_space,bins_range=bins_range,hog_channel=hog_channel)
    notcar_features=f_utils.extract_features(not_cars,color_space=color_space,bins_range=bins_range,hog_channel=hog_channel)

    #3- convert cars_features list and notcars_features list to a vector of features
    # to be used for training
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    #print(X.shape)                       

    #4- Scale the features vector using "scale_vector" from "classifier_utils"
    X_scaler=cl_utils.scale_vector(X)
    
    
    scaled_X=X_scaler.transform(X)
    #print(scaled_X.shape)

    #5- Get the labels vector (1 for cars and 0 for notcars)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    #print(y.shape)

    #6- Split the data set to training set and test set and shuffle their contents
    # for that, "get_train_test_set" from "classifier_utils" is used
    x_train,x_test,y_train,y_test=cl_utils.get_train_test_set(scaled_X=scaled_X,y=y,test_size=0.1)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    #7- Training a classifier using "classifier_fit" from "classifier_utils"
    svc=cl_utils.classifier_fit(x_train,y_train,classifier='linear_svm')

    #8- check the accuracy of a classifier using "classifier_accuracy" from "classifier_utils"
    print("The accuracy of the classifier is {}".format(cl_utils.classifier_accuracy(x_test,y_test,svc)))

    pickle.dump(svc,open( "svc.p", "wb" ))
    pickle.dump(X_scaler,open("X_scaler.p","wb"))

if __name__ == '__main__':
	classify()
