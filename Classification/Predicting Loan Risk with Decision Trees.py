# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:19:40 2020

# This script uses a decision tree classifier to predict if a loan is risky
or safe based on the inputs of 12 features. The script then uses a confusion 
matrix to determine the number of false positive and false negatives of the 
loan risk classifier model.
"""

import pandas as pd
import json 
import matplotlib as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

loans = pd.read_csv('lending-club-data.csv')

# bad_loans column - in this column 1 means a risky (bad) loan 0 means a safe loan
# change the label to 1 means safe loan and -1 means bad loan

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop(['bad_loans'], axis = 1)


# the data has 67 variables for predicting loan risk - use only a subset of the
# the variables

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
X = loans[features]
y = loans[target]

# Apply one-hot encoder to each column with categorical data
categorical_variables = []
for feat_name, feat_type in zip(X.columns, X.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_loans = pd.DataFrame(OH_encoder.fit_transform(X[categorical_variables]))
OH_cols_loans.columns = OH_encoder.get_feature_names()
OH_cols_loans.index = X.index


X = X.drop(categorical_variables, axis = 1)
X = pd.concat([X, OH_cols_loans], axis=1)

# split the data
with open('module-5-assignment-1-train-idx.json') as f:
  train_idx = json.load(f)
  
with open('module-5-assignment-1-validation-idx.json') as f:
  validation_idx = json.load(f)  

train_data = X.iloc[train_idx]
y_train = y[train_idx]
validation_data = X.iloc[validation_idx, :]
y_valid = y[validation_idx]

# Build decision trees with max depth of 2 and 6
small_model = DecisionTreeClassifier(max_depth=2)
small_model.fit(train_data, y_train.to_numpy())

depth6_model = DecisionTreeClassifier(max_depth=6)
depth6_model.fit(train_data, y_train.to_numpy())

# looking at prediction accurary

# concat validation_data and y_valid
joined_validation_data = pd.concat([validation_data, y_valid], axis=1)
validation_safe_loans = joined_validation_data[joined_validation_data[target] == 1]
validation_risky_loans= joined_validation_data[joined_validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

predicted_values = depth6_model.predict(sample_validation_data.iloc[:, :-1])
predict_probabilties = depth6_model.predict_proba(sample_validation_data.iloc[:, :-1])

# predict outcomes and probabilties with the small model
small_predict_probs = small_model.predict_proba(sample_validation_data.iloc[:, :-1])
small_prediction = small_model.predict(sample_validation_data.iloc[:, :-1])


# compute the accurary of the model with a depth of 6 and depth 2

depth_6_accuracy = depth6_model.score(validation_data,y_valid)
depth_2_accuracy = small_model.score(validation_data,y_valid)


# computing oppertunity costs of mistakes in the model through looking at the
# false positives and negatives
predictions = depth6_model.predict(validation_data)

plot_confusion_matrix(depth6_model,
                      validation_data, 
                      y_valid,
                      cmap=plt.cm.Blues) 
confusion_matrix(y_valid, predictions)


