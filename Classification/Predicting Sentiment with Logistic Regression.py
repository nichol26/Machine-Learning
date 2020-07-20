# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:32:07 2020

This script predicts sentiment from product reviews by building a binary
classifier using sparse matrixes and logistic regression. The script then 
compares the accurary of the predictions of a model that includes all the 
words in the reviews as weights and a simpler model that only includes 
predetermined important words. Lastly as a baseline quality check the model is 
compared to a majority classifier.


"""

import pandas as pd
import numpy as np
import json
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

products = pd.read_csv('amazon_baby.csv')

# function to remove punctuation from review column 
def remove_punctuation(text):
    table = str.maketrans('', '', string.punctuation)
    stripped = text.translate(table) 
    return stripped

# fill NA values
products = products.fillna({'review':''})

# remove punctuation from review column
products['review_clean'] = products['review'].apply(lambda x: remove_punctuation(x))


# ignore ratings = since they are considered neutral. Drop them from the dataframe
products = products[products['rating'] != 3]

# classify sentiment column as +1 if rating is greater than 3 else -1
products['sentiment'] = products['rating'].apply(lambda rating : +1
                                                 if rating > 3 else -1)

# Upload list of indices for training and test data
with open('module-2-assignment-train-idx.json') as f:
  train_idx = json.load(f)
with open('module-2-assignment-test-idx.json') as g:
    test_idx = json.load(g)


train_products = products.iloc[train_idx, :]
test_products = products.iloc[test_idx, :]

# provided lines of code - create sparse matrices of word count
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
     # Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_products['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_products['review_clean'])

# create a logistic regression model
sentiment_model = LogisticRegression().fit(train_matrix, 
                                            train_products['sentiment'])

# Count the number of non zero coefficients
coef_array = sentiment_model.coef_
# count non negative coefficients
np.count_nonzero(coef_array >= 0)

# test the function with just a few test points
sample_test_data = test_products.iloc[[10,11,12], :]

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)

# code to predict class labels of sample_test_data based on decision function
def predict_label(scores):
    class_label = []
    y = 0
    for s in scores:
        if s > 0 :
            y = 1
        else:
            y = -1
        class_label.append(y)
    return class_label

# use scores and a sigmoid functions to determine the probability of a positive label
def sigmoid_function(score):
    probability_positive_label = 1/(1 + np.exp(-1*score)) 
    return probability_positive_label

for s in scores:
    print(sigmoid_function(s))

# find highest and loweest 20 scores for the entire test product dataset
test_product_scores = sentiment_model.decision_function(test_matrix)

test_products['scores'] = test_product_scores
largest_scores_20 = test_products.nlargest(20, 'scores')
lowest_scores_20 = test_products.nsmallest(20, 'scores')

# determine the accurary of the sentiment_model
# 1. predict class labels
test_predictions = sentiment_model.predict(test_matrix)
# 2. compute teh accurary
correct_samples = accuracy_score(test_products['sentiment'], test_predictions)

# -------------------Training a simpler model ------------------------


# only use a significant words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_products['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_products['review_clean'])

# train a logistic model
simple_sentiment_model = LogisticRegression().fit(train_matrix_word_subset, 
                                            train_products['sentiment'])
# Put coefficients in a dataframe
simple_model_coef_table = pd.DataFrame({'word':significant_words,
                                         'coefficient':simple_sentiment_model.coef_.flatten()})

simple_model_coef_table.sort_values(by=['coefficient'])

# comparing the accuracy of the two models
train_predictions = sentiment_model.predict(train_matrix)
train_accuracy = accuracy_score(train_products['sentiment'], train_predictions)


# simple model
simple_train_predictions = simple_sentiment_model.predict(train_matrix_word_subset)
simple_train_accuracy =  accuracy_score(train_products['sentiment'], simple_train_predictions)
simple_test_predictions = simple_sentiment_model.predict(test_matrix_word_subset)
simple_test_accuracy = accuracy_score(test_products['sentiment'], simple_test_predictions)



# comparing the sentiment_model performance to a majority classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_model = dummy_clf.fit(train_matrix,train_products['sentiment'])
dummy_predictions = dummy_model.predict(test_matrix)
dummy_accuracy = accuracy_score(test_products['sentiment'], dummy_predictions)
