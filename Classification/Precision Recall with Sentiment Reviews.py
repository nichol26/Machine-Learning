# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:33:31 2020

@author: Sarah
"""

import pandas as pd
import string
import json 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# ignore rating equal to 3
products = products[products['rating'] != 3]


''' ssign reviews with a rating of 4 or higher to be positive reviews, 
while the ones with rating of 2 or lower are negative. For the sentiment 
column, we use +1 for the positive class label and -1 for the negative class label.'''

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

# Upload list of indices for training and test data
with open('module-9-assignment-train-idx.json') as f:
  train_idx = json.load(f)
with open('module-9-assignment-test-idx.json') as g:
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

# inspect the accuaracy 
accuracy = accuracy_score(y_true=test_products['sentiment'].to_numpy(),
                          y_pred=sentiment_model.predict(test_matrix))
print("Test Accuracy: %s" % accuracy)

# look at the baseline accuracy
baseline = len(test_products[test_products['sentiment'] == 1])/len(test_products)
print("Baseline accuracy (majority class classifier): %s" % baseline)

# look at the confusion matrix of the sentiment model
from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true=test_products['sentiment'].to_numpy(),
                        y_pred=sentiment_model.predict(test_matrix),
                        labels=sentiment_model.classes_) 

print(' target_label | predicted_label | count ')
print('--------------+-----------------+-------')
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(sentiment_model.classes_):
    for j, predicted_label in enumerate(sentiment_model.classes_):
        print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
        
# compute the precision and recall
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_true=test_products['sentiment'].to_numpy(), 
                            y_pred=sentiment_model.predict(test_matrix))
print("Precision on test data: %s" % precision)

recall = recall_score(y_true=test_products['sentiment'].to_numpy(),
                      y_pred=sentiment_model.predict(test_matrix))
print("Recall on test data: %s" % recall)


# Exploring tradeoffs between precision and recall


def apply_threshold(probs, threshold):
   sentiment = probs[0].apply(lambda x: +1 if x > threshold else -1)
   return sentiment

probabilities = sentiment_model.predict_proba(test_matrix)[:,1]


probabilities = pd.DataFrame(probabilities)
sentiment_trsh90 = apply_threshold(probabilities, .9)
sentiment_thrsh50 = apply_threshold(probabilities, .5)

# compute precision and recall for threshold of .9
y_true =test_products['sentiment'].to_numpy()
precision = precision_score(y_true, 
                            sentiment_trsh90.to_numpy())
print("Precision on test data: %s" % precision)

recall = recall_score(y_true,
                      sentiment_trsh90)
print("Recall on test data: %s" % recall)


# Precision and Recall Curve
threshold_values = np.linspace(0.5, 1, num=100)
print(threshold_values)

precision_values =[]
recall_values = []

for t in threshold_values:
    sentiment_list = apply_threshold(probabilities,t)
    precision_values.append(precision_score(y_true, sentiment_list))
    recall_values.append(recall_score(y_true, sentiment_list))
    
precision_values = np.delete(precision_values, -1)
recall_values = np.delete(recall_values, -1)
threshold_values = np.delete(threshold_values,-1)
    
def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    
plot_pr_curve(precision_values, recall_values, 'Precision recall curve (all)')

threshold_values[precision_values >=.965]

# find number of false positives in threshold value = .98
sentiment_trsh98 = apply_threshold(probabilities, .98)


cmat = confusion_matrix(y_true,
                        sentiment_trsh98,
                        labels=sentiment_model.classes_) 

print(' target_label | predicted_label | count ')
print('--------------+-----------------+-------')
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(sentiment_model.classes_):
    for j, predicted_label in enumerate(sentiment_model.classes_):
        print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
        
        
#Precision-Recall only on reviews with the word baby
def x_contains(word,string):
    if word in string:
        return True
    else: 
        return False
    
test_string = "testing testing I'm just suggesting"
    
        
baby_reviews = test_products[test_products['review_clean'].apply(lambda x: x_contains('baby', x))]
baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities_baby = pd.DataFrame(sentiment_model.predict_proba(baby_matrix)[:,1])


baby_precision_values = []
baby_recall_values = []
threshold_values = np.linspace(0.5, 1, num=100)




for t in threshold_values:
    baby_sentiment_list = apply_threshold(probabilities_baby,t)
    baby_precision_values.append(precision_score(baby_reviews['sentiment'],baby_sentiment_list))
    baby_recall_values.append(recall_score(baby_reviews['sentiment'],baby_sentiment_list))
    
baby_precision_values = np.delete(baby_precision_values, -1)
baby_recall_values = np.delete(baby_recall_values, -1)


plot_pr_curve(baby_precision_values, baby_recall_values, 'Precision recall curve (all)')

threshold_values = np.delete(threshold_values,-1)
threshold_values[baby_precision_values >=.965]