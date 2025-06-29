import pandas as pd
from scipy.stats import skew

def choose_outlier_method(df, skew_thresh=1.0):
    # Determine the skewness of numeric columns in the DataFrame
    numeric_cols = df.select_dtypes(include='number').columns
    method_map = {}
    # Iterate through numeric columns and determine the method based on skewness
    for col in numeric_cols:
        col_skew = skew(df[col].dropna())
        if abs(col_skew) > skew_thresh:
            method_map[col] = 'iqr'
        else:
            method_map[col] = 'zscore'
    return method_map