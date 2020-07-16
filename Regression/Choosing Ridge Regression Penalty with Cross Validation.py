# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:08:06 2020


This script first analyses how setting the magnitude of the alpha
for the l2 penalty term in ridge regression can reduce the variability of the 
model. This script then uses cross validation to choose an alpha for the l2 penalty term 
that minimizes the residual sum of squares. 


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


# this function takes in a feature and a desired degree and produces a new 
# ddataframe with the desired degree of polynomial
def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            poly_dataframe[name] = feature**power 
    return poly_dataframe


# function that uses the linear model with ridge regression from scikit learn
# library to fit a ridge regression model
def ridge_regression(X, y, penalty):
    model = linear_model.Ridge(alpha=penalty, normalize=True)
    model.fit(X,y)
    predicted_output = model.predict(X)
    num_vars = X.shape[1]
    model_coeffs= model.coef_.reshape(num_vars,1)
    df_model_coeffs = pd.DataFrame(model_coeffs, X.columns, 
                               columns=['Coefficient'])
    return df_model_coeffs, predicted_output

# this function takes in a file name, calls the polynomial_dataframe function
# and then call the ridge_regression function 
def predictions(dataset_title, dictionary, X_feature, y, order, penalty):
    dataset = pd.read_csv(dataset_title, dtype=dictionary)
    dataset = dataset.sort_values(by=[X_feature, y])
    dataset_poly = polynomial_dataframe(dataset[X_feature], order)
    coeffs, predictions = ridge_regression(dataset_poly,
                                             dataset[y], penalty)
    return(dataset,dataset_poly, predictions, coeffs)


# This function splits data into k folds and conducts cross validation from scratch
def k_fold_cross_validation(k, l2_penalty, data, output):
    rss_sum = 0
    n = data.shape[0]
    for i in range(k):
        start = round(((n*i)/k))
        end = round(((n*(i+1))/k))
        valid_set_X = data[start:end+1]
        valid_set_y = output[start:end+1]
        training_set_X = data[0:start].append(data[end+1:n])
        training_set_y = output[0:start].append(output[end+1:n])
        model = linear_model.Ridge(alpha=l2_penalty, normalize=True)
        model.fit(training_set_X,training_set_y)
        predicted_output = model.predict(valid_set_X)
        residuals = valid_set_y - predicted_output
        rss = sum(residuals * residuals)
        rss_sum += rss
    avg_validaton_error = rss_sum/k
    return(avg_validaton_error)

# ----OBSERVING THE EFFECTS OF CHANGING ALPHA OF THE L2 PENALTY -------------

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 
              'floors':str, 'condition':int, 'lat':float, 'date':str, 
              'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 
              'view':int}

# Upload data
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(by =['sqft_living', 'price'])


# Assign a penalty variable for ridge regression
l2_small_penalty = 1.5e-5


# Create a polynomial of order 15 and perform a ridge regression to obtain
# coefficients
poly15_data = polynomial_dataframe(sales['sqft_living'], 15) 
model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_data, sales['price'])
model_coeffs= model.coef_.reshape(15,1)
df_model_coeffs = pd.DataFrame(model_coeffs, poly15_data.columns, 
                               columns=['Coefficient'])



# test ridge regression function
test_coefs, test_output = ridge_regression(poly15_data, sales['price'],l2_small_penalty)

# Download more datasets 

# Repeat for each of the datasets
list_setdata = ['wk3_kc_house_set_1_data.csv', 'wk3_kc_house_set_2_data.csv',
                'wk3_kc_house_set_3_data.csv','wk3_kc_house_set_4_data.csv' ]

# Reassign penalty variable
l2_small_penalty=1e-9

# contruct a for loop that loops through the list of dataset to be loaded and returns
# coefficients
i = 1
data_dict = {}
poly_data_dict = {}
predictions_dict = {}
coeffs_dict = {}

for set_title in list_setdata:
    data, poly_data, predics, coeffs = predictions(set_title,  dtype_dict, 'sqft_living', 
                                                   'price', 15,l2_small_penalty )
    data_title = 'set' + str(i)  
    poly_data_title = 'polyset' + str(i)
    predictions_title = str(i) + 'predictions'
    coeffs_title = 'coeffs' + str(i)
    
    data_dict[data_title] = data['price']
    poly_data_dict[poly_data_title] = poly_data
    predictions_dict[predictions_title] = predics
    coeffs_dict[coeffs_title] = coeffs
    i+=1


# For each of the dictionary values create seperate dataframes:

for key,val in data_dict.items():
        exec(key + '=val')
 
for key,val in poly_data_dict.items():
        exec(key + '=val')   

predictions1 = predictions_dict['1predictions']
predictions2 = predictions_dict['2predictions']
predictions3 = predictions_dict['3predictions']
predictions4 = predictions_dict['4predictions']


for key,val in coeffs_dict.items():
        exec(key + '=val')       
        

# plot the predictions to see how they vary depending on dataset
plt.plot(polyset1['power_1'],set1,'.',
polyset1['power_1'], predictions1,'-')

plt.plot(polyset2['power_1'],set2,'.',
polyset2['power_1'], predictions2,'-')

plt.plot(polyset3['power_1'],set3,'.',
polyset3['power_1'], predictions3,'-')

plt.plot(polyset4['power_1'],set4,'.',
polyset4['power_1'], predictions4,'-')


# see how assigning a large penalty reduces the variability of the model
l2_large_penalty=1.23e2

lpset1_coeffs, lpset1_output = ridge_regression(polyset1, set1, l2_large_penalty)
lpset2_coeffs, lpset1_output = ridge_regression(polyset2, set2, l2_large_penalty) 
lpset3_coeffs, lpset1_output = ridge_regression(polyset3, set3, l2_large_penalty)
lpset4_coeffs, lpset1_output = ridge_regression(polyset4, set4, l2_large_penalty)


#------ CHOOSING ALPHA FOR L2 PENALTY USING CROSS VALIDATION -----------------

# upload and prepare data
train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
poly_X = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)
y = train_valid_shuffled['price']



# Run through a number of potential l2penalties
RSS_penalty_dict = {}

# Use k_fold_cross_validation function to determine l2 penalty that produces
# lowest average residual sum of squares during cross validation
for l2penalty in np.logspace(1, 7, num=13):
    dict_key = 'RSS for Penalty ' + str(l2penalty)
    RSS_penalty_dict[dict_key] = k_fold_cross_validation(10, l2penalty, poly_X, y)
    
# print the minimum RSS penalty
min(RSS_penalty_dict.items(), key=lambda x: x[1])
    
