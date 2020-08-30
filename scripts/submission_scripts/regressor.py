import os
import shutil
import pickle
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from functools import partial


# custom modules
from pipelines import (
    get_pipeline_selectors,
    get_postproc_pipeline,
    load_pickled_encodings,
)

############################################################
############### first let's define the model ###############
############################################################
def tilted_loss(y, f, q=0.9):
    """q is which quantile are we predicting (0.9 = 90th percentile)"""
    e = y - f
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)


def create_dense_regressor(n_input_dims, quantile_regression=False):
    """
    Creates and returns model architecture for a dense network with dropout
    Input(520)  --> Dense(1024) --> droput(0.5) --> Dense(512) --> dropout(0.5) -->Dense(256) -->dropout(0.5) -->Dense(1)
    """

    input_shape = n_input_dims
    input_features = Input(shape=input_shape)

    # model arch
    x = Dense(
        1024,
        activation="relu",
    )(input_features)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    fvc_prediction = Dense(1)(x)

    regressor = Model(input_features, fvc_prediction)
    if quantile_regression:
        regressor.compile(loss=tilted_loss, optimizer="adam")
    else:
        regressor.compile(loss="LogCosh", optimizer="adam")

    # regressor.summary()

    return regressor


###########################################
################ lets train ###############
###########################################
def train_model(model, X, y, validation_split, suffix=None, n_epochs=10):
    now = datetime.datetime.now().isoformat(timespec="minutes")
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'./models/regressor_model_{"" if suffix is None else suffix}_{now}',
        monitor="val_loss",
        save_best_only=True,
    )
    tensorboard_callback = TensorBoard(log_dir="/tmp/regressor")

    model.fit(
        X,
        y,
        epochs=n_epochs,
        batch_size=32,
        shuffle=True,
        validation_split=validation_split,
        callbacks=[tensorboard_callback, model_checkpoint_callback],
    )