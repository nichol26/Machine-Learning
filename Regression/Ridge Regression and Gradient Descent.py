# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:58:59 2020

@author: Sarah
"""

# This script tunes regression parameters by using an l2 penalty and
# gradient descent 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1  
    features = ['constant'] + features # this is how you combine two lists
    features_frame = data_frame[features]
    feature_matrix = features_frame.to_numpy()
    output_array = data_frame[output]
    output_array = output_array.to_numpy()
    return(feature_matrix, output_array)

# Matrix of predictions
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

# Function to compute derivative of the ridge regression
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
   if feature_is_constant:
      derivative = np.sum(2*(errors*feature)) 
   else:
      derivative = np.sum(2*(errors*feature)) + 2*l2_penalty*weight
   return derivative

# Function that performs gradient descent on ridge regression
def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, 
                                      step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights) 
    itr = 1
       
    while itr <= max_iterations:
        # compute the predictions using your predict_output() function
        predicts = predict_outcome(feature_matrix, weights)
        errors = predicts - output
        for i in range(len(weights)): 
        
            if i == 0:
                feature_is_constant = True
            else:
                feature_is_constant = False
            derivative = feature_derivative_ridge(errors, 
                                                  feature_matrix[:,i], weights[i], 
                                                  l2_penalty, feature_is_constant)
            weights[i] = weights[i] - (step_size*derivative)
        itr+=1
    return weights


# Upload data
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 
              'floors':str, 'condition':int, 'lat':float, 'date':str, 
              'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 
              'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype = dtype_dict)


# Testing the predicted_outcome function and the feature_derivative_ridge function
example_features, example_output = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_outcome(example_features, my_weights)
errors = test_predictions - example_output 



# next two lines should print the same values
print(feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False))
print(np.sum(errors*example_features[:,1])*2+20)


# next two lines should print the same values
print(feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True))
print(np.sum(errors)*2)



# ----- PUT TITLE HERE
test_data = pd.read_csv('kc_house_test_data.csv', dtype = dtype_dict) 
train_data = pd.read_csv('kc_house_train_data.csv', dtype = dtype_dict) 

simple_features = ['sqft_living']
my_output = 'price'
simple_feature_matrix, output = get_numpy_data(train_data, simple_features, my_output)
simple_test_feature_matrix, test_output = get_numpy_data(test_data, simple_features, my_output)

# Exlpore how different penalty sizes affect the magnitude of the trained parameters
# from gradient descent
initial_weights = np.array([0., 0.])
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix,
                                                             output, initial_weights,
                                                             1e-12, 0, 1000)
    
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix,
                                                             output, initial_weights,
                                                             1e-12, 1e11, 1000)
    
# plot the learning of the variables
plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_outcome(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_outcome(simple_feature_matrix, simple_weights_high_penalty),'r-')



#Looking at RSS on the test data for 0 penalty
test_predictions_0 = predict_outcome(simple_test_feature_matrix, simple_weights_0_penalty)
test_RSS_0_Penalty = np.sum((test_predictions_0 - test_output)**2)

#Looking at RSS on the test data for high penalty
test_predic_hp = predict_outcome(simple_test_feature_matrix, simple_weights_high_penalty)
test_RSS_high_penalty = np.sum((test_predic_hp - test_output)**2)


# Exploring ridge regression penalties and RSS with 2 input features
initial_mweights = np.array([0., 0., 0.])
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
feature_matrix, output = get_numpy_data(train_data, model_features, my_output)
test_feature_matrix, test_output = get_numpy_data(test_data, model_features, my_output)

multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                             output, initial_mweights,
                                                             1e-12, 0, 1000)

multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                             output, initial_mweights,
                                                             1e-12, 1e11, 1000)
#Looking at RSS on the test data for 0 penalty
mtest_predictions_0 = predict_outcome(test_feature_matrix, multiple_weights_0_penalty)
mtest_RSS_0_penalty = np.sum((mtest_predictions_0 - test_output)**2)

#Looking at RSS on the test data for high penalty
mtest_predictions_hp = predict_outcome(test_feature_matrix, multiple_weights_high_penalty)
mtest_RSS_high_penalty = np.sum((mtest_predictions_hp - test_output)**2)