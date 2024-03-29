"""Post processing functions for the OSIC Pulmonary Fibrosis Kaggle Challenge"""
import pickle
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
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

def get_postproc_pipeline(no_op_attrs:list=[], num_attrs:list=[], cat_attrs:list=[], encoded_attrs:list=[]):
    """Returns a post-processing pipeline for a unified DF"""
    ops = []
    # Define the no-operation pipeline
    no_op_pipeline = Pipeline([
        ('selector', DataFrameSelector(no_op_attrs)),
    ])
    ops.append(no_op_pipeline)
    # Define the numerical pipeline
    if len(num_attrs) > 0:
        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attrs)),
            ('imputer', SimpleImputer(strategy="median")),
            # Hardcoded to the range of the test set
            # ('std_scaler', MinMaxScaler(feature_range=(-12, 133))),
            ('std_scaler', StandardScaler()),
        ])
        ops.append(num_pipeline)
    # Define the categorical pipeline
    if len(cat_attrs) > 0:
        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attrs)),
            ('one_hot_encoder', OneHotEncoder()),
        ])
        ops.append(cat_pipeline)
    # Define the encoded CT column pipeline
    if len(encoded_attrs):
        encoded_pipeline = Pipeline([
            ('selector', DataFrameSelector(encoded_attrs)),
            ('imputer', SimpleImputer(strategy="median")),
        ])
        ops.append(encoded_pipeline)
    return make_union(*ops)


def get_pipeline_selectors():
    """Returns lists of column names to use with get_postproc_pipeline()"""
    # WARN: This is a helper function that must be tweaked by hand on every change
    # No-op weeks because time progress is significant?
    no_op_attrs = []
    num_attrs = ["Weeks", "Age", "TRAPZ_VOL"]
    cat_attrs = ["Sex", "SmokingStatus"]
    # After the join, the encoded CT columns have integer column labels between 0 & n - 1
    # Where n is the dimensions of the CT bottleneck encoding: n = 500;
    encoded_attrs = []

    return no_op_attrs, num_attrs, cat_attrs, encoded_attrs
