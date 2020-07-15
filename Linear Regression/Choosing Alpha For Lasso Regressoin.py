# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:37:27 2020

"""
'''
This script explores how different values of alpha for a linear regression model
that includes an l1 penalty terms effects the number of parameters and the residual
sum of squares of the predicted outputs in the validation dataset. This script
also uses the validation set to select an alpha that reduces the number of parameters
in the model to 7 and out of the models with 7 parameters the one that has the lowest
residual sums of squares
'''

import numpy as np
import pandas as pd
from math import sqrt
from sklearn import linear_model

# dowload data
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']


testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)


training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

# find the appropriate alpha for the l1 penalty term using the training a validation data
RSS_dict = {}
for l1_penalty in np.logspace(1,7,num=13): # loop through potential l1 penalties
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training['price'])
    predictions = model.predict(validation[all_features])
    RSS = np.sum((predictions - validation['price'])**2)
    RSS_dict[str(l1_penalty)] = RSS
    
# print the minimum RSS penalty
min(RSS_dict.items(), key=lambda x: x[1])

# use the alpha for the l1 penalty term that produces the minimum RSS in the validation data to 
# fit training data
# in this case alpha = 10 produces the lowest RSS
model_l1opt = linear_model.Lasso(alpha=10, normalize=True) # set parameters
model_l1opt.fit(training[all_features], training['price'])

# look at the coefficient values with an alpha = 10 
num_vars = len(all_features)
model_coeffs_l1opt= model_l1opt.coef_.reshape(num_vars,1)
df_model_coeffs = pd.DataFrame(model_coeffs_l1opt, all_features, 
                               columns=['Coefficient'])

# Count the number of non zero coefficients
l1opt_nonzero_weights = (np.count_nonzero(model_l1opt.coef_) + 
                        np.count_nonzero(model_l1opt.intercept_))


# Imagine desired number of coefficients to be included in the model was 7
# Find an alpha for the l1_penalty term to set max number of non-zero coefficients to 7
model_coeffs_list =[]
column_names_list = []

# Loop through alpha values for the l1 penalty term to find range of alphas that
# produce models with the number of parameters equal to 7
for l1_penalty in np.logspace(1,4,num=20):
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training['price'])
    non_zero_coefficients = (np.count_nonzero(model.coef_) + 
                             np.count_nonzero(model.intercept_))
    column_name = str(l1_penalty)
    model_coeffs_list.append(non_zero_coefficients)
    column_names_list.append(column_name)
    
non_zero_weights_df = pd.DataFrame(
    {'l1 penalty value': column_names_list,
     'non zero weights': model_coeffs_list
     
     })


# Narrowing down the range for max number of non zero weights = 7 and pick the 
# alpha with the lowest RSS
p_model_coeffs_list =[]
p_column_names_list = []
p_RSS_list = []

for l1_penalty in np.linspace(127,264,20):
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training['price'])
    predictions = model.predict(validation[all_features])
    RSS = np.sum((predictions - validation['price'])**2)
    non_zero_coefficients = (np.count_nonzero(model.coef_) + 
                             np.count_nonzero(model.intercept_))
    column_name = str(l1_penalty)
    p_model_coeffs_list.append(non_zero_coefficients)
    p_column_names_list.append(column_name)
    p_RSS_list.append(RSS)
    
max_weights_df = pd.DataFrame(
    {'l1 penalty value': p_column_names_list,
     'non zero weights': p_model_coeffs_list,
     'RSS values': p_RSS_list
     })

# desired alpha value for l1 penalty term is located at coordinates 4,0 in the
# max_weights_df 
# Use this chosen_l1penalty value to train a linear model on the training data
# that will only contain seven parameters   
chosen_l1penalty = float(max_weights_df.iloc[4,0])
opt_model = linear_model.Lasso(alpha=chosen_l1penalty, normalize=True)
opt_model.fit(training[all_features], training['price'])
num_vars = len(all_features)
model_coeffs= opt_model.coef_.reshape(num_vars,1)
df_model_coeffs = pd.DataFrame(model_coeffs, all_features, 
                               columns=['Coefficient'])