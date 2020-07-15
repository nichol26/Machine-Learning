# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 09:15:57 2020

"""

'''
 This script uses coordinate descent with a linear regression model that includes
 a lasso term as the regularization parameter. To carry out coordinate descent
 with a linear model with a lasso term this script uses functions that 
1. turn data into numpy arrays
2. normalize the data
3. predict housing prices
4. carries out coordinate descent for the ith weight
5. carries out coordinate descent for all weights until max of the absolute values
   in the change of the weight in neglibly small
'''

import pandas as pd
import numpy as np

def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1  
    features = ['constant'] + features # this is how you combine two lists
    features_frame = data_frame[features]
    feature_matrix = features_frame.to_numpy()
    output_array = data_frame[output]
    output_array = output_array.to_numpy()
    return(feature_matrix, output_array)

def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return (normalized_features, norms)

# Matrix of predictions
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    prediction = predict_outcome(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.sum(feature_matrix[:,i]*((output - prediction) +
                   weights[i]*feature_matrix[:,i]))
    
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.
    
    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output,
                                      initial_weights, l1_penalty, tolerance):
    converged = False
    change = np.array(initial_weights)*0
    weights = np.array(initial_weights)
    
    while not converged:
                       
        for i in range(len(initial_weights)):
            new_weight = lasso_coordinate_descent_step(i, feature_matrix, 
                                                       output, weights, l1_penalty)
            change[i] = np.abs(new_weight - weights[i])
            weights[i] = new_weight
        if max(change) < tolerance:
            converged = True
    
    return weights

# --EXPLORING EFFECTS OF COORDINATE DESCENT AND LASSO ON HOUSING DATA -----
    

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# download data and turn into numpy arrays
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
simple_features_matrix, simple_output = get_numpy_data(sales, 
                                                       ['sqft_living', 'bedrooms'],
                                                       'price')

# normalize data
simple_normalized_features, simple_norms = normalize_features(simple_features_matrix)
init_weights = np.array([1,4,1])
simple_predictions = predict_outcome(simple_normalized_features, init_weights)

# carry out coordinate descent until convergence
initial_weights = [0,0,0]
computed_weights = lasso_cyclical_coordinate_descent(simple_normalized_features, 
                                                     simple_output,
                                      initial_weights, 1e7, 1)

# compute RSS using the computed weights
simple_predictions = predict_outcome(simple_normalized_features, computed_weights)
simple_RSS = np.sum((simple_output - simple_predictions)**2)


# exploring the effects of lasso with more features

# dowload training set and turn data into numpy arrays
training_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
multi_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                  'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                  'sqft_basement','yr_built', 'yr_renovated']

multi_feature_matrix, multi_feature_output = get_numpy_data(training_data, 
                                                            multi_features, 'price')
# normalize data
multi_normalized_features, multi_norms = normalize_features(multi_feature_matrix)
initial_weights = np.zeros(14)

# Carry out coordinate descent
multi_weights = lasso_cyclical_coordinate_descent(multi_normalized_features, 
                                                     multi_feature_output,
                                      initial_weights, 1e7, 1)
final_list = ['intercept'] + multi_features
non_zero_feature_list = pd.DataFrame({'feature': final_list,
                                     'feature weight' : multi_weights})


# increase the lasso 
multi_weights = lasso_cyclical_coordinate_descent(multi_normalized_features, 
                                                     multi_feature_output,
                                      initial_weights, 1e8, 1)

final_list = ['intercept'] + multi_features
non_zero_feature_list = pd.DataFrame({'feature': final_list,
                                     'feature weight' : multi_weights})



# decrease the lasso
multi_weights = lasso_cyclical_coordinate_descent(multi_normalized_features, 
                                                     multi_feature_output,
                                      initial_weights, 1e4, 1)
final_list = ['intercept'] + multi_features
non_zero_feature_list = pd.DataFrame({'feature': final_list,
                                     'feature weight' : multi_weights})

