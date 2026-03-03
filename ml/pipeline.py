"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class GPATrendTransformer(BaseEstimator, TransformerMixin):
    """
    Compute mean GPA and GPA trend (slope) across available semester gpa columns.
    Expects columns named sem1_gpa ... sem8_gpa (missing values allowed).
    Adds 'mean_sem_gpa' and 'gpa_trend' features and drops the sem#_gpa columns by default.
    """
    def __init__(self, sem_cols=None, keep_sem_cols=False):
        self.sem_cols = sem_cols or [f"sem{i}_gpa" for i in range(1,9)]
        self.keep_sem_cols = keep_sem_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        def slope(row):
            vals = []
            xs = []
            for i, c in enumerate(self.sem_cols, start=1):
                v = row.get(c, np.nan)
                if not pd.isna(v):
                    xs.append(i)
                    vals.append(v)
            if len(vals) < 2:
                return 0.0
            # simple linear fit slope
            m, _ = np.polyfit(xs, vals, 1)
            return float(m)
        X["mean_sem_gpa"] = X[self.sem_cols].mean(axis=1)
        X["gpa_trend"] = X.apply(slope, axis=1)
        if not self.keep_sem_cols:
            X = X.drop(columns=[c for c in self.sem_cols if c in X.columns])
        return X
"""