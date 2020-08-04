# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:56:34 2020

@author: Sarah
"""

import pandas as pd
import numpy as np
import json
import matplotlib as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

loans = pd.read_csv('lending-club-data.csv')

# bad_loans column - in this column 1 means a risky (bad) loan 0 means a safe loan
# change the label to 1 means safe loan and -1 means bad loan

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop(['bad_loans'], axis = 1)
target = 'safe_loans'
loans = loans[features + [target]]

# fill na values with none in the emp_length column
for column in loans.columns:
    loans[column] = loans[column].fillna('none')
    
# Apply one-hot encoder to each column with categorical data
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_loans = pd.DataFrame(OH_encoder.fit_transform(loans[categorical_variables]))
OH_cols_loans.columns = OH_encoder.get_feature_names()
OH_cols_loans.index = loans.index


loans = loans.drop(categorical_variables, axis = 1)
loans = pd.concat([loans, OH_cols_loans], axis=1)


# split the data
with open('module-8-assignment-2-train-idx.json') as f:
  train_idx = json.load(f)
  
with open('module-8-assignment-2-test-idx.json') as f:
  validation_idx = json.load(f)  


train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx, :]


''' ---- CONTRUCTING A WEIGHTED DECISION TREE ----------------------------'''

def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    # Sum the weights of all entries with label +1
    total_weight_positive = sum(data_weights[labels_in_node == +1])
    
    # Weight of mistakes for predicting all -1's is equal to the sum above
    ### YOUR CODE HERE
    weighted_mistakes_all_negative = total_weight_positive
    
    # Sum the weights of all entries with label -1
    ### YOUR CODE HERE
    total_weight_negative = sum(data_weights[labels_in_node == -1])
    
    # Weight of mistakes for predicting all +1's is equal to the sum above
    ### YOUR CODE HERE
    weighted_mistakes_all_positive = total_weight_negative
    
    # Return the tuple (weight, class_label) representing the lower of the two weights
    #    class_label should be an integer of value +1 or -1.
    # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
    ### YOUR CODE HERE
    if weighted_mistakes_all_negative < weighted_mistakes_all_positive:
        
        return weighted_mistakes_all_negative, -1
    elif weighted_mistakes_all_positive < weighted_mistakes_all_negative:
        
        return  weighted_mistakes_all_positive, 1
    else:
        return weighted_mistakes_all_positive,1
    

# If the data is identical in each feature, this function should return None

def best_splitting_feature(data, features, target, data_weights):
    
    # These variables will keep track of the best feature and the corresponding error
    best_feature = None
    best_error = float('+inf') 
    num_points = float(len(data))

    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        # The right split will have all data points where the feature value is 1
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        
        # Apply the same filtering to data_weights to create left_data_weights, right_data_weights
        ## YOUR CODE HERE
        left_data_weights = data_weights[data[feature] ==0]
        right_data_weights = data_weights[data[feature] ==1]
                    
        # DIFFERENT HERE
        # Calculate the weight of mistakes for left and right sides
        ## YOUR CODE HERE
        left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
        right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)
         # DIFFERENT HERE
        # Compute weighted error by computing
        #  ( [weight of mistakes (left)] + [weight of mistakes (right)] ) / [total weight of all data points]
        ## YOUR CODE HERE
        error = (left_weighted_mistakes + right_weighted_mistakes)/ (np.sum(left_data_weights) + np.sum(right_data_weights)) 
        
        # If this is the best error we have found so far, store the feature and the error
        if error < best_error:
            best_feature = feature
            best_error = error
    
    # Return the best feature we found
    return best_feature


def create_leaf(target_values, data_weights):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'is_leaf': True}
    
    # Computed weight of mistakes.
    # Store the predicted class (1 or -1) in leaf['prediction']
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    
    return leaf

def weighted_decision_tree_create(data, features, target, data_weights, current_depth = 1, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    
    # Stopping condition 1. Error is 0.
    if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
        print("Stopping condition 1 reached." )               
        return create_leaf(target_values, data_weights)
    
    # Stopping condition 2. No more features.
    if remaining_features == []:
        print("Stopping condition 2 reached." )               
        return create_leaf(target_values, data_weights)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth > max_depth:
        print("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values, data_weights)
     # If all the datapoints are the same, splitting_feature will be None. Create a leaf
    splitting_feature = best_splitting_feature(data, features, target, data_weights)
    remaining_features.remove(splitting_feature)
        
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    
    print("Split on feature %s. (%s, %s)" % (\
              splitting_feature, len(left_split), len(right_split)))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target], data_weights)
    if len(right_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(right_split[target], data_weights)
    
    # Repeat (recurse) on left and right subtrees
    left_tree = weighted_decision_tree_create(
        left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
    right_tree = weighted_decision_tree_create(
        right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


def classify(tree, x, annotate = False):   
    # If the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction'] 
    else:
        # Split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)



def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    predictions_list = []
    for r in range(data.shape[0]):
        prediction = classify(tree, data.iloc[r,1:], annotate = False)
        predictions_list.append(prediction)
    
    # Once you've made the predictions, calculate the classification error
    return (prediction != data[target]).sum() / float(len(data))

# -------------Looking at the affects of boosting --------------------
example_data_weights =np.concatenate((np.ones(10), 
                                      np.zeros(train_data.shape[0]-20), 
                                      np.ones(10)), axis = None)

features_list = list(train_data.columns)
features_list.remove('safe_loans')


'''

small_data_decision_tree_subset_20 = weighted_decision_tree_create(train_data, 
                                                                   features_list, 
                                                                   'safe_loans', 
                                                                   example_data_weights,
                                                                   current_depth = 1, 
                                                                   max_depth = 2)


evaluate_classification_error(small_data_decision_tree_subset_20, train_data)

                   
row_3 = train_data.iloc[3, :]
classify(small_data_decision_tree_subset_20, row_3, annotate = False)

'''

from math import log
from math import exp

def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    data = data.reset_index(drop=True)
    # start with unweighted data
    alpha = np.ones(len(data))
    alpha_pd = pd.DataFrame(np.ones(data.shape[0]))
    weights = []
    tree_stumps = []
    target_values = data[target]
    predictions = []
    
    for t in range(num_tree_stumps):
        print('=====================================================')
        print('Adaboost Iteration %d' % t)
        print('=====================================================')        
        # Learn a weighted decision tree stump. Use max_depth=1
        tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, max_depth=1)
        tree_stumps.append(tree_stump)
        
        # Make predictions
        
        for r in range(data.shape[0]):
            prediction = classify(tree_stump, data.iloc[r,1:], annotate = False)
            predictions.append(prediction)
        
        # Produce a Boolean array indicating whether
        # each data point was correctly classified
        is_correct = predictions == target_values
        is_wrong   = predictions != target_values
        predictions.clear()
        # Compute weighted error
        # YOUR CODE HERE
        
        
        wrongs = alpha_pd[is_wrong]
        talpha = np.sum(alpha_pd)
        sum_wrong = np.sum(wrongs)
        
        
        weighted_error = np.sum(alpha_pd[is_wrong])/ np.sum(alpha_pd)
         # Compute model coefficient using weighted error
        # YOUR CODE HERE
        weight =.5*np.log((1-weighted_error)/weighted_error)
        weights.append(weight)
        
        # Adjust weights on data point
        adjustment = is_correct.apply(lambda is_correct : exp(-weight) if is_correct else exp(weight))
        
        # Scale alpha by multiplying by adjustment
        # Then normalize data points weights
        ## YOUR CODE HERE 
        alpha = alpha*adjustment
        alpha = alpha/np.sum(alpha)
    
    return weights, tree_stumps


subset_data = train_data.iloc[0:100,:]


weights_10_itr, tree_stumps_10_itr  = adaboost_with_tree_stumps(train_data, 
                                                features_list, 
                                                'safe_loans', 
                                                10)


def predict_adaboost(stump_weights, tree_stumps, data):
    stump_weights_array = np.array(stump_weights)
    prediction_matrix = np.zeros((data.shape[0], len(tree_stumps)))
    for i, tree_stump in enumerate(tree_stumps):
       
        
        for r in range(data.shape[0]):
            prediction = classify(tree_stumps, data.iloc[r,1:], annotate = False)
            prediction_matrix[r][i] = prediction
            
    scores_vector = np.dot(stump_weights_array.transpose(), prediction_matrix)
    return scores_vector.apply(lambda score : +1 if score > 0 else -1)

# Looking at the effects of adding stumps
    
# train a model with 30 stumps
weights_30_itr, tree_stumps_30_itr  = adaboost_with_tree_stumps(train_data, 
                                                features_list, 
                                                'safe_loans', 
                                                30)


error_all = []
for n in range(1, 31):
    predictions = predict_adaboost(weights_30_itr[:n], tree_stumps_30_itr[:n], train_data)
    error = 1.0 - accuracy_score(train_data[target], predictions)
    error_all.append(error)
    print("Iteration %s, training error = %s" % (n, error_all[n-1]))
    
# Plotting training error versus number of tree stumps

plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.legend(loc='best', prop={'size':15})

plt.rcParams.update({'font.size': 16})
      
# Plotting training and validaton error versus number of tree stumps
validation_error_all = []
for n in range(1, 31):
    predictions = predict_adaboost(weights_30_itr[:n], tree_stumps_30_itr[:n], validation_data)
    error = 1.0 - accuracy_score(validation_data[target], predictions)
    error_all.append(error)
    print("Iteration %s, training error = %s" % (n, error_all[n-1]))
    

plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.plot(range(1,31), validation_error_all, '-', linewidth=4.0, label='Test error')

plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.rcParams.update({'font.size': 16})
plt.legend(loc='best', prop={'size':15})
plt.tight_layout()