# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4) 

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

---
## This project include a [writeup](writeup.md) file that contains the steps used to implement this project.
## The projects includes some modules that contain some functions and variables. These modules are organized as follows:
* [color_utils](color_utils.py): for raw intensity features and histogram features extraction.
* [gradient_utils](gradient_utils.py): for HOG feature extraction.
* [windows_utils](windows_utils.py): for making sliding windows and drawing bounding boxes
* [feature_utils](features_utils.py):for different features extraction (color and HOG)
* [global_variabls](globals_variables.py):for the variables used in features extraction to be set once and for the whole project.
* [classifier_utils](classifier_utils.py):for normalizing and classification issues.
* [multi_detections_utils](multi_detections_utils.py): to handle multi detections and false positives issues.
beside these modules, there are two other top view models that used all of the previous ones.
* [classify](classify.py): Used for training a classifier
* [main](main.py): Used as a pipeline to implement the project.
---
For more details, please have a look at [writeup](writeup.md) file or you can check the final video from [here](video.mp4).   