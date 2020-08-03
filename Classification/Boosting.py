# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 09:20:30 2020

@author: Sarah
"""
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

loans = pd.read_csv('lending-club-data.csv')

# bad_loans column - in this column 1 means a risky (bad) loan 0 means a safe loan
# change the label to 1 means safe loan and -1 means bad loan

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop(['bad_loans'], axis = 1)


# keep only a subset of the data
target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]


loans = loans[[target] + features]

# inspect data for missing values
loans.info()


# split the data
with open('module-6-assignment-train-idx.json') as f:
  train_idx = json.load(f)
  
with open('module-6-assignment-validation-idx.json') as f:
  validation_idx = json.load(f)  

train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx, :]

# drop na values
train_data = train_data.dropna()
validation_data = validation_data.dropna()


# encode catagorical data for training and validation
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)

train_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(train_encoder.fit_transform(train_data[categorical_variables]))
OH_cols_train.columns = train_encoder.get_feature_names()
OH_cols_train.index = train_data.index

train_data = train_data.drop(categorical_variables, axis = 1)
train_data = pd.concat([train_data, OH_cols_train], axis=1)

valid_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_valid = pd.DataFrame(valid_encoder.fit_transform(validation_data[categorical_variables]))
OH_cols_valid.columns = valid_encoder.get_feature_names()
OH_cols_valid.index = validation_data.index

validation_data = validation_data.drop(categorical_variables, axis = 1)
validation_data = pd.concat([validation_data, OH_cols_valid], axis=1)




# train a gradient boosting classifier
boosting_model_n5 = GradientBoostingClassifier(n_estimators = 5, max_depth = 6)
boosting_model_n5.fit(train_data.iloc[:, 1:], train_data.iloc[:, 0])

# predict if loans are risky or not
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

sample_predictions = boosting_model_n5.predict(sample_validation_data.iloc[:,1:])
sample_probs = boosting_model_n5.predict_proba(sample_validation_data.iloc[:,1:])

# looking at model accuracy for entire validation test
predictions = boosting_model_n5.predict(validation_data.iloc[:,1:])

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(boosting_model_n5,validation_data.iloc[:,1:] ,
                      validation_data.iloc[:,0])

from sklearn.metrics import confusion_matrix
n5_confusion_matrix = confusion_matrix( validation_data.iloc[:,0], predictions)

# finding the safest loans
probabilities = boosting_model_n5.predict_proba(validation_data.iloc[:,1:])
validation_data['good_loans'] = probabilities[:,0]
validation_data['bad_loans'] = probabilities[:,1]

sorted_good = validation_data.sort_values(by=['good_loans'])
sorted_bad = validation_data.sort_values(by=['bad_loans'])

validation_data = validation_data.drop(['good_loans', 'bad_loans'], axis = 1)

# comparing esemble models with different numbers of boosting stages
number_boosting_stages = [10,50,100,200,500]

accuracy_scores = {}
for i in number_boosting_stages:
    boosting_model = GradientBoostingClassifier(n_estimators = i, max_depth = 6)
    boosting_model.fit(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    predictions = boosting_model.predict(validation_data.iloc[:,1:])
    accuracy =  accuracy_score(validation_data.iloc[:,0], predictions)
    accuracy_scores[str(i)] = accuracy 
    

    

    
