"""
This module contains functions related to classification process (car - noncar).
It contains the following functions:
		1- scale_vector	 		:to scale a feature vector
		2- get_train_test_set	:to split a data set to training set and testing set
		3- train 				:a function that used to train a specific classifier
		4- classifier_accuray	:a function that used to get the accuracy of a specific classifier on the test set
		5- classifier_predict	:a function that used to get a label prediction given features vector 
		
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #sklearn>=0.18
from sklearn.svm import LinearSVC
import numpy as np

#---------------------------------------------------------------------------------------------#
"""
# 1- scale_vector

This function is used to scale a feature vector to be used later to train a classifier.
The function uses StandardScaler from sklearn.preprocessing .
input: unscaled feature vector X.
output: a scaled featured vector scaled_X.
"""

def scale_vector(X):
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)
	return scaled_X

#--------------------------------------------------------------------------------------------#

"""
# 2- get_train_test_set

This function is used to split the whole data set to a training set and a test set.
The function uses train_test_split from sklearn.model_selection (for sklearn>=0.18)
If (sklearn)=0.17, train_test_split can be imported from sklearn.cross_validation.
The function also shuffle the data randomly.

inputs:		- scaled_X 		: scaled feature vectors
			- y 			: labels
			- test_size		: the portion of the test set in the whole data set (0.2 is suitable)
			- random_state 	: a random number that used to shuffle the data

outputs:	- x_train, y_train 	: training data set contains features and labels respectively
			- x_test, y_test	: testing data set contains features and labels respectively  
"""

def get_train_test_set(scaled_X,y,test_size=0.2,random_state=np.random.randint(0, 100)):
	
	x_train, x_test,  y_train, y_test=train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
	
	return x_train,x_test,y_train,y_test
#--------------------------------------------------------------------------------------------------------#

"""
# 3- train

This function is used to train a specific classifier on the training set.

inputs: 	x_train,y_train : training features and their labels
			classifier		: the classifier that will be used

output: 	a trained_classifier
"""
def train(x_train,y_train,classifier='linear_svm'):

	if classifier=='linear_svm':
		svc=LinearSVC()
		svc.fit(x_train,y_train)
		return svc

#--------------------------------------------------------------------------------------------#

"""
# 4- classifier_accuracy

This function is used to get the accuracy of a specific classifier on the test set
inputs: 	x_test,y_test	: features and labels of the test sets
			classifier 		: The classifier that used during training

outputs: 			 		: the accuracy of the classifier on tested on the test set
"""

def classifier_accuracy(x_test,y_test,classifier='linear_svm'):
	if classifier=='linear_svm':
		return svc.score(x_test, y_test)

#--------------------------------------------------------------------------------------------#

"""
# 5- classifier_predict

This function is used to get the accuracy of a specific classifier on the test set
inputs: 	x_test			: vector of features of  images
			classifier 		: The classifier that used during training

outputs: 	 		 		: predicted label of these features
"""

def classifier_accuracy(x,classifier='linear_svm'):
	if classifier=='linear_svm':
		return svc.predict(x)

	 	 
