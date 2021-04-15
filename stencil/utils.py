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

pd.set_option('display.max_rows', 1000)
pd.options.display.max_columns = 500
pd.options.display.width = 1000


colors = {'crimson': '#a50026',   'red': '#d73027',
          'redorange': '#f46d43', 'orange': '#fdae61',
          'yellow': '#fee090',    'sky': '#e0f3f8',
          'babyblue': '#abd9e9',  'lightblue': '#74add1',
          'blue': '#4575b4',      'purple': '#313695'}


def _summary(df):
    d = pd.DataFrame()
    cols = df.columns
    for c in cols:
        d[c] = [df[c].isnull().sum(), df[c].isnull().sum()/df.shape[0]*100,
                len(df[c].unique()), len(df[c].unique())/df.shape[0]*100,
                [df[c].unique()]]
    d = d.T
    d.columns = ['#null', '%null', '#unique', '%unique', 'unique_values']
    return d

def summary_all(df):
    return pd.concat([df.describe().T[['count', 'min', 'max', 'mean']], summary_(df)], axis=1, sort=False)


# Convert null or missing values to np.nan
def df_normalize_strings(df, in_cols=[], out_cols=[], na_value=np.nan):
    if in_cols==[]: in_cols = df.columns
    for col in df[in_cols].drop(out_cols, axis=1).columns:
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            df[col] = df[col].str.lower()
            df[col] = df[col].fillna(na_value)  
            df[col] = df[col].replace('none', na_value)
            df[col] = df[col].replace('', na_value)

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

# make histogram with dataframe and a list of columns
def histogram(df, col=[], t=np.int, bins=5, cmin=-np.inf, cmax=np.inf,
              title=None, xlabel=None, ylabel=None):
    s = df[col].values.astype(t)
    n, bin, patches = plt.hist(s.clip(min=cmin, max=cmax), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.show()

def rmse(y_pred, y_hat): 
    return np.sqrt(np.mean((y_pred-y_hat)**2))

