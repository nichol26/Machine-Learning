# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:17:04 2020

@author: Sarah
"""

''' This script containts a series of functions that 
1. turn data into numpy arrays
2. normalize the data
3. compute euclidean distances between obersvations
4. find the k nearest neihghbours to a chosen datapoint
5. Based on the k nearest neighbours predict the price house of a chosen observation
   point
6. Predict a series of housing prices based on k nearest neighbours

This script then uses the functions to select a value of k that produces the lowest
RSS between the computed housing prices and the housing prices in the validation
data. It then uses the optimal number of k nearest neighbours to predict the housing
prices in the test dataset
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

def compute_distances(features_instances, features_query):
    distances = np.sqrt(np.sum((features_instances - features_query)**2, axis = 1))
    return distances


def k_nearest_neighbors(k, feature_train, features_query):
    distances = compute_distances(feature_train, features_query)
    neighbors = np.argsort(distances) 
    neighbors = neighbors[0:k]
    return neighbors


def predict_output_of_query(k, features_train, output_train, features_query):
    k_neighbors = k_nearest_neighbors(k, features_train, features_query)
    prediction = np.mean(output_train[k_neighbors])
    return prediction


def predict_output(k, features_train, output_train, features_query):
    predictions = []
    for i in range(features_query.shape[0]):
        prediction = predict_output_of_query(k, features_train, 
                                             output_train, features_query[i,:])
        predictions.append(prediction)
    return predictions
 
# upload datasets
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

feature_list = ['bedrooms',  
                'bathrooms',  
                'sqft_living',  
                'sqft_lot',  
                'floors',
                'waterfront',  
                'view',  
                'condition',  
                'grade',  
                'sqft_above',  
                'sqft_basement',
                'yr_built',  
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']


sales_train = pd.read_csv('kc_house_data_small_train.csv', dtype=dtype_dict)
sales_validation = pd.read_csv('kc_house_data_small_validation.csv', dtype=dtype_dict)
sales_test = pd.read_csv('kc_house_data_small_test.csv', dtype=dtype_dict)

train_features, output_train = get_numpy_data(sales_train, feature_list, 'price')
validation_features, output_validation = get_numpy_data(sales_validation, feature_list, 'price')
test_features, output_test = get_numpy_data(sales_test, feature_list, 'price')

# normalize data
norm_features_train, norms = normalize_features(train_features)
norm_features_valid = validation_features / norms
norm_features_test = test_features / norms



# find a value of k that produces the lowest RSS with the validation set
rss_column_names_list = []
rss_for_k_list = []
for k in range(1,16):
    predictions = predict_output(k, norm_features_train, 
                                 output_train, norm_features_valid)
    RSS = np.sum((output_validation - predictions)**2)
    rss_for_k_list.append(RSS)
    rss_column_names_list.append(str(k))


RSS_for_given_k = pd.DataFrame({'k value': rss_column_names_list,
                                'rss value': rss_for_k_list})



# a value of k = 8 produces the lowest RSS in the validation data
# use this value of k to make predictions on the test data then calculate the RSS
k8_test_predictions = predict_output(8, norm_features_train, 
                                 output_train, norm_features_test)
k8_RSS_test = np.sum((k8_test_predictions - output_test)**2)


