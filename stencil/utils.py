import os
import time
import datetime
import pdb

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype, is_object_dtype
from pandas.api.types import is_categorical_dtype, is_datetime64_dtype
from pandas.api.types import is_int64_dtype, is_float_dtype

from matplotlib import pyplot as plt

# feature importance
from rfpimp import *

# set pandas options
pd.set_option('display.max_rows', 1000)
pd.options.display.max_columns = 500
pd.options.display.width = 1000

# universal colors
colors = {'crimson'  : '#a50026', 'red'      : '#d73027',
          'redorange': '#f46d43', 'orange'   : '#fdae61',
          'yellow'   : '#fee090', 'sky'      : '#e0f3f8',
          'babyblue' : '#abd9e9', 'lightblue': '#74add1',
          'blue'     : '#4575b4', 'purple'   : '#313695'}


############################
####### SANITY CHECK #######
############################

def sanity_check(df):
    for col in df.columns:
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            print(f"Col {col} is still a string")
        if df[col].isnull().any():
            print(f"Col {col} still has missing values")

def check_types(df1,df2):
    if df1.shape[1] != df2.shape[1]:
        print(f"Num columns differs: {df1.shape[1]} != {df2.shape[1]}")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    if cols1 != cols2:
        print(f"Column names differ:")
        if len(cols1-cols2)>0:
            print(f"\tIn df1 not df2: {cols1-cols2}")
        if len(cols2-cols1)>0:
            print(f"\tIn df2 not df1: {cols2-cols1}")
    for col in cols1.intersection(cols2): # check those in common
        if df1[col].dtype != df2[col].dtype:
            print(f"Col {col} dtypes differ {df1[col].dtype} != {df2[col].dtype}")

# sanity_check(X)
# sanity_check(X_valid)
# check_types(X, X_valid)


###########################
### SUMMARY INFORMATION ###
###########################

def _summary(df):
    d = pd.DataFrame()
    cols = df.columns
    for c in cols:
        d[c] = [df[c].isnull().sum(), df[c].isnull().sum()/df.shape[0]*100,
                len(df[c].unique()), len(df[c].unique())/df.shape[0]*100,
                df[c].unique()]
    d = d.T
    d.columns = ['#null', '%null', '#unique', '%unique', 'unique_values']
    return d

def summary_all(df):
    dtype = pd.DataFrame({'type': df.dtypes})
    return pd.concat([df.describe().T[['count', 'min', 'max', 'mean']], 
                      dtype, _summary(df)], axis=1, sort=False)


###########################
##### FILLING MISSING  ####
###########################

# Fill missing numberical values by median & adding one more 
# dummy column whose values indicate where the null values are
def fix_missing_num(df, cols, value=None):
    for col in cols:
        df[col+'_na'] = pd.isnull(df[col])
        if value is None: df[col].fillna(df[col].median(), inplace=True)
        else: df[col].fillna(value, inplace=True)

def fix_missing_num_only(df, cols, value=None):
    for col in cols:
        if value is None: df[col].fillna(df[col].median(), inplace=True)
        else: df[col].fillna(value, inplace=True)

# Convert null or missing values to np.nan
def df_normalize_strings(df, in_cols=[], out_cols=[], na_value=np.nan):
    if in_cols==[]: in_cols = df.columns
    for col in df[in_cols].drop(out_cols, axis=1).columns:
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            df[col] = df[col].str.lower()
            df[col] = df[col].fillna(na_value)  
            df[col] = df[col].replace('none', na_value)
            df[col] = df[col].replace('', na_value)

# Extract numeric from column and replace null with np.nan
def extract_numeric(df, col):
    df[col] = df[col].str.extract(r'([0-9.]*)', expand=True)
    df[col] = df[col].replace('', np.nan)
    df[col] = pd.to_numeric(df[col])


#####################################
### ENCODING + FEATURE GENERATION ###
#####################################

# Convert string to category and replace missing value
def df_string_to_cat(df, in_cols=[], out_cols=[]):
    if in_cols==[]: in_cols=df.columns
    for col in df[in_cols].drop(out_cols, axis=1).columns:
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            df[col] = df[col].astype('category').cat.as_ordered()
            
# Add 1 to catcode to offset the value -1 of np.nan
def df_cat_to_catcode(df, in_cols=[], out_cols=[]):
    if in_cols==[]: in_cols=df.columns
    for col in df.drop(out_cols, axis=1).columns:
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes + 1  

# Frequency encoding
def frequency_encode(df, col, new_col=None):
    _count = df[col].value_counts()
    if not new_col: 
        new_col = col + '_freq'
    df[new_col] = df[col].map(_count) 

# Extract features from string
def word_features(df, col, string_list=None):
    if string_list is not None:
        for word in string_list:
            df[word] = df[col].str.contains(word)

# Length-of-text features
def text_len_features(df, col):
    df[col+'_num'] = df[col].apply(lambda x: len(x.split(",")))

# One-hot encoding
def onehot(df, col):
    ascat = df[col].astype('category').cat.as_ordered()
    onehot = pd.get_dummies(df[col], prefix=col, dtype=bool)
    del df[col]
    df = pd.concat([df, onehot], axis=1)
    # return altered dataframe and column training categories
    return df, ascat.cat.categories

# Target encoding
# https://mlbook.explained.ai/catvars.html

# Embeddings

# External data information


####################
## Visualization ###
####################

# make histogram with dataframe and a list of columns
def histogram(df, col=[], dtype=np.int, bins=5, cmin=-np.inf, cmax=np.inf,
              color=colors['blue'], title=None, xlabel=None, ylabel=None):
    s = df[col].values.astype(dtype)
    n, bin, patches = plt.hist(s.clip(min=cmin, max=cmax), 
                                bins=bins, color=color)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.show()

def histogram_percentage(df, col=[], dtype=np.int, bins=5, 
                         upper_bound=1, lower_bound=99, colors['blue'], 
                         title=None, xlabel=None, ylabel=None)
    # e.g. find middle 98% of a column
    upper, lower = np.percentile(df[col], [upper_bound, lower_bound]) 
    clipped = np.clip(df.price, upper, lower)
    fig,ax = plt.subplots()
    ax.hist(clipped, bins=bins, color=colors['blue'])
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.show()


##############
## Metrics ###
##############

def MAE(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def MSE(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def RMSE(y_pred, y_true):
    return np.sqrt(MSE(y_pred, y_true))



# model = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
def test(X, y, model):
    model.fit(X, y)
    oob = model.oob_score_
    nodes = rfnnodes(model)
    height = np.median(rfmaxdepths(model))
    print(f"OOB R^2 {oob:.5f} using {nodes:,d} tree nodes with {height} median tree height")
    return rf, oob


