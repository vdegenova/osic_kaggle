"""Post processing functions for the OSIC Pulmonary Fibrosis Kaggle Challenge"""
import pickle
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_union, Pipeline

# Define custom transformers
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Selects columns from a Pandas DataFrame using attr"""
    def __init__(self, attr: list):
        self.attr = attr

    def fit(self, X, y=None):
        """No-op: nothing to fit"""
        return self

    def transform(self, X):
        """Selects from X the columns listed in self.attr"""
        # X must be a pd.DataFrame
        return X[self.attr]

def load_pickled_encodings(pickle_path, csv_path, key="Patient"):
    """Loads pickled CT encodings and joins them df on 'key'"""
    with open(pickle_path, "rb") as fio:
        patient_ct_encodings = pickle.load(fio)
    # Load the dict with keys as rows & values as columns
    patient_ct_encodings = pd.DataFrame.from_dict(patient_ct_encodings, orient="index")
    # Load the csv dataframe
    data_frame = pd.read_csv(csv_path)
    # Left join to data_frame on key
    return data_frame.join(patient_ct_encodings, on=key, how="left")

def get_postproc_pipeline(no_op_attrs, num_attrs, cat_attrs, encoded_attrs):
    """Returns a post-processing pipeline for a unified DF"""
    # Define the no-operation pipeline
    no_op_pipeline = Pipeline([
        ('selector', DataFrameSelector(no_op_attrs)),
    ])
    # Define the numerical pipeline
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attrs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    # Define the categorical pipeline
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attrs)),
        ('one_hot_encoder', OneHotEncoder()),
    ])
    # Define the encoded CT column pipeline
    encoded_pipeline = Pipeline([
        ('selector', DataFrameSelector(encoded_attrs)),
        ('imputer', SimpleImputer(strategy="median")),
    ])

    return make_union(no_op_pipeline, num_pipeline, cat_pipeline, encoded_pipeline)

def get_pipeline_selectors():
    """Returns lists of column names to use with get_postproc_pipeline()"""
    # WARN: This is a helper function that must be tweaked by hand on every change
    # No-op weeks because time progress is significant?
    no_op_attrs = []
    num_attrs = ["Weeks", "Age"]
    cat_attrs = ["Sex", "SmokingStatus"]
    # After the join, the encoded CT columns have integer column labels between 0 & n - 1
    # Where n is the dimensions of the CT bottleneck encoding: n = 500;
    encoded_attrs = list(range(0, 500))

    return no_op_attrs, num_attrs, cat_attrs, encoded_attrs
