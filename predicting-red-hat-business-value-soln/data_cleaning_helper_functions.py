# Data Cleansing and preprocessing logic
import warnings
warnings.filterwarnings('ignore')

# Data Folders
import os
import sys

# Data Wrangling
import math
import random
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import statistics
from scipy import stats
import calendar
import datetime as dt
from itertools import combinations

# Data Export
import pickle

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# Check column alignment 
def uncommon_elements(list1, list2):
    
    ''' This function checks unique and common elements between lists '''
    
    return [element for element in list2 if element not in list1]

# Duplicated Entires
def duplicated_indicies(data):
    
    ''' This function returns the number of row index for duplicated elements '''
    
    return data[data.duplicated()].index

# Correct Data Types
def correct_dates(data, cols):
    
    ''' This function corrects date type variables'''
    
    return pd.to_datetime(data[cols], format = '%Y-%m-%d')


def correct_objects(data, cols):
    
    ''' This function corrects object type variables'''
    
    return data[cols].astype('object')


def correct_string(data, cols):
    
    ''' This function corrects string type variables'''
    
    return data[cols].astype(str)


def correct_floats(data, cols):

    ''' This function corrects float type variables'''
    
    return data[cols].astype('float')


def correct_category_datatype(data, cols):
    
    ''' This function corrects category type variables'''
    
    return data[cols].astype('category')


def correct_int_datatype(data, cols):
    
    ''' This function corrects int type variables'''
    
    return data[cols].astype('int')


# Missing Value Treatment
def identify_missing_values(train_data):

    ''' This function identifies the amount of missing data per variable'''
    
    print('Nan values =', train_data.isnull().sum().sum())
    print("""""")

    vars_with_missing = []

    for feature in train_data.columns:
        missings = train_data[feature].isna().sum()

        if missings > 0 :
            vars_with_missing.append(feature)
            missings_perc = missings / train_data.shape[0]

            print('Variable {} has {} records ({:.2%}) with missing values.'.format(feature, missings, missings_perc))
    print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
    
    
def plot_missing_data(train_data, test_data):
    
    ''' This function plots the distriubtion of missing data in training and testing data '''
    
    df_missing_train = pd.DataFrame({'column':train_data.columns, 'missing(%)':((train_data.isna()).sum()/train_data.shape[0])*100})
    df_missing_test = pd.DataFrame({'column':test_data.columns, 'missing(%)':((test_data.isna()).sum()/test_data.shape[0])*100})

    df_missing_train_nl = df_missing_train.nlargest(7, 'missing(%)')
    df_missing_test_nl = df_missing_test.nlargest(7, 'missing(%)')

    sns.set_palette(sns.color_palette('nipy_spectral'))

    plt.figure(figsize=(16,6))
    sns.barplot(data= df_missing_train_nl, x='column', y='missing(%)',palette='nipy_spectral')
    plt.title('Missing values (%) in training set')
    plt.show()

    plt.figure(figsize=(16,6))
    sns.barplot(data= df_missing_test_nl, x='column', y='missing(%)',palette='nipy_spectral')
    plt.title('Missing values (%) in test set')
    plt.show()
    

def remove_missing_values(data, thresold_limit):
    
    ''' This function removes columns which have missing values over a predetermined threshold '''
    
    return data.loc[:, data.isnull().sum() < thresold_limit*data.shape[0]]


def missing_value_ratio(data, threshold):
    
    ''' This function removes columns which have missing values over a predetermined threshold '''
    
    missing_data = data.isnull().sum()/len(train_data)*100
    
    variables = data.columns
    variable = []
    dropped_cols = []
    for i in range(0, len(train_data.columns)):
        if missing_data[i] <= threshold:   
            variable.append(variables[i])
        else:
            dropped_cols.append(variables[i])
    
    return variable, dropped_cols


#def impute_blank_missing_values(data, cols):
#    
#    ''' Missing Value imputation example '''
#    
#    return np.where((data[cols] == 'T') | (data[cols] == ' '), -99999, data[cols])


def impute_nan_corrupt_missing_values_numeric(data, cols):
    
    ''' Missing Value imputation example '''
    
    return np.where((data[cols] == '#VALUE!') | (data[cols] == 'nan' | (data[cols] == ' '), -99999, data[cols]))

                    
def impute_nan_corrupt_missing_values_categorical(data, cols):
    
    ''' Missing Value imputation example '''
    
    for i in cols:
        data[i] = np.where(data[i].isna(), "unclassified", data[i])
    
    return data
    
    #return np.where((data[cols] == '#VALUE!') | (data[cols] == 'nan') | (data[cols] == 'NaN') | (data[cols] == ' '), 'unclassified', data[cols])


def impute_negative_missing_data(data):
    
    ''' Replace -1 missing values with nan '''
    
    return data.replace(-999999, np.nan)


# Data Preparation for Columns
def column_overlap(train_data, test_data):
    
    ''' This column checks what columns exist in training but no testing data '''
    
    #print(test_data.columns.difference(train_data.columns)) # Solution One
    print('Columns in train and not in test dataset:', set(train_data.columns) - set(test_data.columns)) # Solution Two
    

def multiple_column_comparison(data, col):   
    
    ''' This function outputs summary statistics for a subset of selectec columns'''
    
    return print(data[col].describe())


def determine_variable_cardinality(data, cols):

    ''' This function outputs the number of unique categories per categorical variable '''
    
    for f in cols:
        dist_values = data[f].value_counts().shape[0]
        print('Variable {} has {} distinct values'.format(f, dist_values))


# Zero-Variance Predictors
def remove_columns_unique_values(data):
    
    ''' This function removes zero variance predictors from the data set in two steps: Faster than zero_variance_predictors  '''
    
    nunique = data.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    
    return data.drop(cols_to_drop, axis=1)


#def zero_variance_predictors(data):
#    
#    ''' This function removes zero variance predictors from the data set '''
#    
#    return data.loc[:, data.apply(pd.Series.nunique) != 1]


def low_variance_filter(data, threshold):
    
    ''' This function identified columns of low variance '''
    
    var = data.var()
    numeric = data.columns
    
    variable = []
    for i in range(0, len(var)):
        if var[i] >= threshold:   
            variable.append(numeric[i+1])
    
    return data[variable]

# Outlier Assessment


# Data Quality and Expections (assert statements)


# Impact of zero


# Normalisation & Scaling 



# Reconciliation 

# Class Imbalance
def determine_class_imbalance(data, col):
    
    ''' This function determines the class imbalace associated with the target class '''
    
    return data[col].value_counts()

##########################################################
# New .py
# New Project Set up
## - Enable multi-cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
## - Warnings
import warnings 
warnings.filterwarnings('ignore')
## - Plotting Import Requirements 


##########################################################
# New .py
# Data Encoding Strategies
# prepare input features
def ordinal_encoding(data):
    ''' Implementation of ordinal encoding '''
    oe = OrdinalEncoder()
    oe.fit(data)
    Ord_enc = oe.transform(data) # Using ordinal variables df
    
    return Ord_enc

# prepare target variable
def label_encoding(y):
    ''' Implementation of label encoding '''
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y) # Using categorial variables df
    
    return y_enc

# One-Hot Encoding
def one_hot_encoding(x, y):
    ''' Implementation of one-hot encoding '''
    #one_hot_df = pd.get_dummies(categorical_feature_eng, prefix='one_hot_ps_ind_02_cat')
    one_hot_df = pd.merge(x, y, left_index=True, right_index=True) # Merge on index -- Safer than concat

    model = linear_regression.LinearRegression()
    model.fit(x[[cols]], y)

    display(model.coef_)
    display(model.intercept_)


# Dummy Encoding
def dummy_encoding(x, y):
    ''' Implementation of dummy encoding '''
    #dummy_df = pd.get_dummies(categorical_feature_eng, prefix=['dummy_ps_ind_02_cat'], drop_first = True)
    dummy_df = pd.merge(x, y, left_index=True, right_index=True) # Merge on index -- Safer than concat

    model = linear_regression.LinearRegression()
    model.fit(x[[cols]], y)

    display(model.coef_)
    display(model.intercept_)


# Hash features for Large Categorical Variables 
## h = FeatureHasher(n_features = m, input_type='string')
## f = h.transform(train_data['ps_car_11_cat'])
## f.toarray()
## print('Our pandas Series, in bytes: ', getsizeof(train_data['ps_car_11_cat']))
## print('Our hashed numpy array, in bytes: ', getsizeof(f))



##########################################################
# New .py
# Categorical Variable Selection

##########################################################
# New .py
# Model 
# Create a benchmark dummy model
random.seed(1234)

def _generate_dummy_model_classification(X, y):
    
    ''' Create dummy models for classification algorithms '''
    
    # Normalisation/Standardisation --- Implement Feature Scaling
    X_scaled = scale(X)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size = 0.2, random_state = 10)

    for strat in ['stratified', 'most_frequent', 'prior', 'uniform']:
        dummy_maj = DummyClassifier(strategy=strat).fit(X_train, y_train)
        print(strat)
        print("Train Stratergy :{} \n Score :{:.2f}".format(strat, dummy_maj.score(X_train, y_train)))
        print("Test Stratergy :{} \n Score :{:.2f}".format(strat, dummy_maj.score(X_train, y_train)))
        print("Cross Validation: Training Data \n ", cross_val_score(dummy_maj, X_train, y_train, cv = 10, scoring = 'accuracy'))
        print("""""")


def _top_predictors(model_data):
    ''' Creation of interaction terms and top 5 predictors '''

    model_data = model_data.dropna()

    y_train = model_data[['MAX']]
    X_train_int = model_data[['Precip', 'Snowfall', 'YR', 'MO', 'DA', 'MIN', 'ELEV', 'ksp_date']]

    columns_list = X_train_int.columns
    interactions = list(combinations(columns_list, 2))

    interaction_dict = {}
    for interaction in interactions:
        X_train_int['int'] = X_train_int[interaction[0]] * X_train_int[interaction[1]]
        interaction_model = LinearRegression(normalize=True)
        interaction_model.fit(X_train_int, y_train)
        interaction_dict[interaction_model.score(X_train_int, y_train)] = interaction

    top_5 = sorted(interaction_dict.keys(), reverse = True)[:5]
    for interaction in top_5:
        print(interaction_dict[interaction])
        
        
def _top_poly_predictors(model_data):
    ''' Creation of polynominal terms and top predictor '''

    poly_dict = {}
    poly_X_train_int = model_data[['Precip', 'Snowfall', 'YR', 'MO', 'DA', 'MIN', 'ELEV', 'ksp_date']]
    for feature in poly_X_train_int.columns:
        for p in range(2, 5):
            X_train_poly = poly_X_train_int
            X_train_poly['sq'] = X_train_poly[feature]**p
            lr = LinearRegression(normalize=True)
            lr.fit(X_train_poly, y_train)
            poly_dict[lr.score(X_train_poly, y_train)] = [feature, p]

    print(poly_dict[max(poly_dict.keys())])


# New .py
# Statistical Hypothesis Testing 


# New .py
# Programme Quality and Assessment of Technical Debt 
## - watermark
## - Progress Bars
## - Time cells to complete

#!pip install ipython-autotime
#%load_ext autotime

## - Parallel Processing
## - Coding Standards & Linting: Black / Mypy / flake8 / Viztracer / MLNotify / Sonar Cube (pycharm)


# New .py
# Results and Output Storage
## - Store data and respective results with suitable timestamp 
def _save_model_to_pickle(file_location, model, model_type):
    
    filename = dt.datetime.now().strftime('%Y-%m-%d')+'_'+model_type+'_model.pkl'
    
    pickle.dump(model, open(file_location+filename, 'wb'))
    
## - Automatically download html version of notebook
## - Ability to quantify quality in the results 
## - Aim Logs (https://github.com/aimhubio/aim)