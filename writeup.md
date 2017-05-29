# **Vehicle Detection Project**

The goals / steps of this project are the following:

* 1-Apply a color transform and append binned color features, as well as histograms of color.
* 2-Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images. 
* 3-Normalize the features from the first two steps and randomize a selection for training and testing and train a classifier Linear SVM classifier.
* 4-Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* 5-Run my pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* 6-Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/car_not_car_color_features.png
[image3]: ./examples/hog_pic.png
[image4]: ./examples/unnormal_normal.png
[image5]: ./examples/window.png
[image6]: ./examples/multi.png
[image7]: ./examples/detected.png


---
## 1- Apply a color transform and append binned color features, as well as histograms of color.
To get color features and histograms of color from an image, we should set some parameters such as image color space, spatial size, histogram bins and bins range. These parameters can modified in once and be used later as global variables in [global variables](globals_variables.py) file. After setting these parameters, color and histogram features can be obtained using "extract_color_features" from [color_utils](color_utils.py). This step should be applied, later, on each of the training images. For example, applying this step on the following two images (car and not car)
![alt text][image1]
will give an output as below
![alt test][image2]
Color features are not robust against orientation. Histogram features are robust against orientation but unfortunately different objects with the same color, for example a red car and a red building will give the same color histogram and we will not able to determine whether it is a car or a building. So, using color features is not enough and we need to use, additionally, gradient features.
## 2- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
As color features, the parameters used for HOG should be assigned once in [global variables](globals_variables.py) file. These parameters are orient, pixels per cell, cell per block, and hog channel. After assigning these parameters the HOG features of an image can be extracted using "extract_hog_features" from [gradient_utils](gradient_utils.py). an example of this gradient is shown below
![alt text][image3]
* For the algorithm, the color features(raw intensities and histogram) and the gradient features are both required. To get them in one vector (raw intensities, histogram and HOG). To get all of them, a "extract_features" from [features_utils](features_utils.py).

## 3-Normalize the features from the first two steps and randomize a selection for training and testing and train a classifier Linear SVM classifier
Before training our classifier, one important step is normalizing. Normalizing is very important in any machine learning algorithm to prevent features with large scale dominate on the classifier during training. To do that the features vector obtained from the previous step is normalized using "scale_vector" function from [classifier_utils](classifier.utils). An example of the features before and after normalization is shown below.
![alt text][image4]
After normalizing the data features, it is time to split it into two sets, training and testing and shuffle them. It can be done using "get_train_test_set" function from [classifier_utils](classifier.utils). To train a Linear SVM classifier, function "classifier_fit" from [classifier_utils](classifier.utils) is used. This function fits an SVM classifier over a training set.
* The previous steps can be done using [classify](classify.py) function. This function loads data set for cars and not cars, extract features from both of them, scale these features and train an SVM classifier. The parameters of this classifier then is stored in [svc.p](svc.p) file using pickle.dump and later can be used by loading them using pickle.load.

## 4-Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
To detect a car the bottom half of the image is divided to number of windows with appropriated size.For me, I used two scales (64x64) and (96x96). At every window a classifier is used to determine whether this window contains a cat or not. An example of such sliding window can be shown below.
![alt text][image5]
To get this sliding window, "slide_windows" function from [windows_utils](windows_utils.py) is used. This function allows to chooses the size of the windows and the overlapping between them.
## 5-Run my pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
All the previous steps can be put in on pipeline and applied once on every frame of a video stream. This pipeline is a function called "find_cars" in [features_utils](features_utils.py). The output of this function is boxes around the places where a car may be detected. The boxes can suffer from two problems, multi detections and false positive. Multi detections is detecting the same object over frames and drawing multi but different sizes boxes around it. False positive, happens when a classifier thinks that a certain spot is a car but in fact it is not. To overcome these two problems, a heat map is used. In brief a heat map is a way in which incremental every detected possible candidate spot by one and after several detection we apply a certain threshold with labels to allow us grouping multi detections in one detection and eliminate false positives. The threshold value is a value that is needed to be tuned. The functions that can be used to get a heat map, labeling it and applying threshold are found in [multi_detections_utils](multi_detections_utils.py).
An example of these multi detections and heat map can be shown below.

![alt text][image6]

## 6-Estimate a bounding box for vehicles detected.
By applying this pipeline on every frame with the heat map will lead to drawing a box around the detected cars as in the following image.

![alt text][image7]

---
# Rubric points of this project
## 1-The image features (color features and HOG)
The color features have been calculated by setting its corresponding parameters in [global variable](globals_variables.py) as follows:
* color_space = 'YCrCb' 
* spatial_size = (32, 32)
* spatial_feat = True
* hist_feat = True
* hist_bins = 32
* bins_range=(0,1)
The HOG has calculated by setting its corresponding parameters in [global variable](globals_variables.py) as follows:
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = "ALL"
* hog_feat = True
These features after that are scaled to zero mean and unit variance using "scale_vector" function in [classifier_utils](calssifier_utils.py). The scaled features are used to train a Linear SVM classifier. "classify" function in [classify.py](classify.py) was used to fit the data and store the classifier's parameters.
## 2-A sliding window was applied with scale 1.5 and window size 64 over the bottom part of the image and then on each window a classify predicted whether it is a car or not. "find_cars" function from [features_utils](features_utils.py) was used for that. The reliability of the classifier was improved by training more data from different data sets (KTTI and GTI) nad by tuning the scale and window size parameters. Also, using [multi_detections_utils](multi_detections_utils.py) helped to improve the prediction and reducing false positives.
## 3-This pipeline was used to detect cars on each frame of frame sequences, the detected candidate areas were used to plot a heat map, then by properly choosing a threshold the multi detection was eliminated and the false positives were reduced.   


---

# Discussion

## 1. One thing I need to improve is to make the detection process more smooth, the boxes appeared and disappeared somehow sharply, it will be good if the boxes change in its size smoothly as the cars go far. 


