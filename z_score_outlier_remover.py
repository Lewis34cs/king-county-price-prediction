import numpy as np
import pandas as pd

class ZScoreOutlierRemover:
    def __init__(self, z_thresh=3.0, columns=None):
        self.z_thresh = z_thresh
        # If columns are not specified, they will be determined from the data
        self.columns = columns
        # Dictionaries to store the means and std devs for each column
        self.means = {}
        self.stds = {}

    def fit(self, X):
        # Convert input to DataFrame if it is not already
        X = pd.DataFrame(X)
        if self.columns is None:
            # If columns are not specified, select numeric columns
            self.columns = X.select_dtypes(include=np.number).columns.tolist()
        valid_cols = []
        for col in self.columns:
            std = X[col].std()
            if std == 0:
                print(f"Warning: column '{col}' has zero std and will be ignored.")
            else:
                valid_cols.append(col)
                # Calculate the mean and standard deviation for each column
                # and store them in the respective dictionaries
                self.means[col] = X[col].mean()
                self.stds[col] = std
        self.columns = valid_cols
        return self

    def transform(self, X, return_mask=True):
        # Convert input to DataFrame if it is not already
        X = pd.DataFrame(X)
        # Create a mask for outliers based on the Z-score method
        mask = pd.Series(True, index=X.index)
        # for each column
        for col in self.columns:
            # Calculate the Z-score for each value in the column
            # and update the mask
            z_score = (X[col] - self.means[col]) / self.stds[col]
            mask &= (z_score.abs() < self.z_thresh)
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
