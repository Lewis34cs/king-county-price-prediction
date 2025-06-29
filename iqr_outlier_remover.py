import pandas as pd
import numpy as np

class IQROutlierRemover:
    def __init__(self, iqr_thresh=1.5, columns=None):
        self.iqr_thresh = iqr_thresh
        # If columns are not specified, they will be determined from the data
        self.columns = columns
        # Dictionaries to store the quantiles and IQR for each column
        self.q1 = {}
        self.q3 = {}
        self.iqr = {}

    def fit(self, X):
        # Convert input to DataFrame if it is not already
        X = pd.DataFrame(X)
        # If columns are not specified, select numeric columns
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()
            # for every column
        for col in self.columns:
            # Calculate the first and third quartiles and the IQR
            # and store them in the respective dictionaries
            self.q1[col] = X[col].quantile(0.25)
            self.q3[col] = X[col].quantile(0.75)
            self.iqr[col] = self.q3[col] - self.q1[col]
        return self

    def transform(self, X, return_mask=True):
        # Convert input to DataFrame if it is not already
        X = pd.DataFrame(X)
        # Create a mask for outliers based on the IQR method
        mask = pd.Series(True, index=X.index)
        # for each column
        for col in self.columns:
            # Calculate the lower and upper bounds for outliers
            # and update the mask
            lower_bound = self.q1[col] - self.iqr_thresh * self.iqr[col]
            upper_bound = self.q3[col] + self.iqr_thresh * self.iqr[col]
            mask &= (X[col] >= lower_bound) & (X[col] <= upper_bound)
        # if y is provided, filter both X and y
        if return_mask:
            return mask
        # if y is not provided, return only X
        else:
            return X.loc[mask]
    
    # Fit the transformer and then transform the data
    def fit_transform(self, X, return_mask=True):
        self.fit(X)
        return self.transform(X, return_mask)