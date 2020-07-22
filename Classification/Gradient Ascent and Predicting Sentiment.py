# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:29:56 2020

This notebook uses gradient ascent of the gradients of a maximum likelihood 
function to train parameters for a classifier model that predicts wether a 
review is positive or negative based on the prescence of prechosen 'important'
words.
"""

import pandas as pd
import numpy as np
from math import sqrt
import string
import json
from sklearn.metrics import accuracy_score

#load data
products = pd.read_csv('amazon_baby_subset.csv')


# function to remove punctuation from review column 
def remove_punctuation(text):
    table = str.maketrans('', '', string.punctuation)
    stripped = text.translate(table) 
    return stripped

# fill NA values
products = products.fillna({'review':''})

# remove punctuation from review column
products['review_clean'] = products['review'].apply(lambda x: remove_punctuation(x))


# only of subset of the words in the reviews are considered 'important'
# upload this subset of words
with open('important_words.json') as f:
  important_words = json.load(f)
  
# compute the number of each of the important words in the review
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
    
# count the number of reviews that contain the words perfect
perfect_mask = products['perfect'] >=1
rows_with_perfect = products[perfect_mask]


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.to_numpy()
    label_array = dataframe[label]
    label_array = label_array.to_numpy()
    return(feature_matrix, label_array)


# use get_numpy_data to extract important words columns and sentiment from products
product_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')


#function that computes input and probability output of a sigmoid function
def predict_probability(feature_matrix, coefficients):
    score = np.dot(feature_matrix, coefficients)
    predictions = 1/(1 + np.exp(-1*score))
    return predictions

# function that computes derivatives of the log likelihood with respect to a 
# single coefficient w_j   
def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors,feature)
        # Return the derivative
    return derivative

# function to compute log likehood of a score being correct
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores)))
    return lp


def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in range(max_iter):
        # Predict P(y_i = +1|x_1,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)

        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions

        for j in range(len(coefficients)): # loop over each coefficient
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
            # compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative(errors, feature_matrix[:,j])

            # add the step size times the derivative to the current coefficient
            coefficients[j] += step_size*derivative
            
        # Checking whether log likelihood is increasing
        if itr % 10 ==0: 
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print(str(itr), lp)
    return coefficients
    
# run a logistic regression and observe the maximum likelihood is increasing
initial_coefficients = np.zeros(194)    
computed_coefficients= logistic_regression(product_matrix, sentiment, 
                                           initial_coefficients, 1e-7, 301)


# predict number of positive sentiment based on scores
scores = np.dot(product_matrix, computed_coefficients)
predicted_sentiment = list(map(lambda x: 1 if x>0 else -1, scores))
positive_sentiment = len(list(filter(lambda x: (x >= 0), predicted_sentiment)))



# compute accurary of the predicted_sentiment
accuracy = accuracy_score(products['sentiment'], predicted_sentiment)


# determine the most important positive word and negative words
coefficients = list(computed_coefficients[1:]) 
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
