import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow.keras.backend as K
import keras.losses


# custom modules
from pipelines import (
    get_pipeline_selectors,
    get_postproc_pipeline,
    load_pickled_encodings,
)


def tilted_loss(y, f):
    q = 0.9
    e = y - f
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)


def load_model(filename="./models/regressor_model_D_0.5_LogCosh"):
    # keras.losses.custom_loss = tilted_loss
    model = keras.models.load_model(
        filename, custom_objects={"tilted_loss": tilted_loss}
    )
    return model


def infer(model, x):
    preds = model.predict(x)
    return preds