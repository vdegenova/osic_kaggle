# playing with efficientnet
# install efficientnet with `pip install -U efficientnet` :)
import numpy as np
import json
import os
import random
import pandas as pd
from typing import Tuple
import efficientnet.tfkeras as efn
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate)
from tensorflow.keras.optimizers import Adam
from myDataGenerator import myDataGenerator
import sys
sys.path.append('./src/')
from pipelines import (
    get_pipeline_selectors,
    get_postproc_pipeline,
    load_pickled_encodings,)

# https://kobiso.github.io/Computer-Vision-Leaderboard/imagenet.html
# B0 expects 224x224
# B1 expects 240x240
# B2 expects 260x260
# B3 expects 300x300
# B4 expects 380x380
# B5 expects 456x456
# B6 expects 528x528
# B7 expects 600x600

def build_wide_and_deep(
    input_shape:Tuple[int,int,int]=(224,224,3),
    weights='noisy-student',
    ):
    '''
    Builds a wide and deep model by concatenating a base efficientnet model --with a vector of tabular data
    INPUTS
        input_shape (Tuple(int,int,int)): input image size. May have only 1 channel.
            Models trained with sizes in chart above. Default is B0
        weights (str): May be one of None, 'imagenet', 'noisy-student', or weights file location
    RETURN
        model
    '''

    # cast input to 3 channels
    inp = Input(shape=input_shape)
    #inp = Conv2D(3, (3, 3), activation="relu", padding="same", input_shape=input_shape)(inp)

    #base_input_shape = (input_shape[0], input_shape[1], 3)
    base = efn.EfficientNetB0(input_shape=input_shape, weights=weights, include_top=False)
    for layer in base.layers:
        layer.trainable = False
    
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1)(x) # activation defaults to linear

    model = Model(inp, x)

    # compile model
    opt = Adam(learning_rate=1e-5)
    loss = 'MSE'
    model.compile(optimizer=opt, loss=loss)

    return model


def load_training_dataset(LOCAL_PATIENT_MASKS_DIR:str, LOCAL_PATIENT_TAB_PATH:str,
    validation_split=0.7):
    # converts a loaded data dict of masked images into a proper dataset for NN training
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    datagen_params = {
        'dim': (224, 224),
        'batch_size': 32,
        'n_channels': 3,
        'data_dir':LOCAL_PATIENT_MASKS_DIR
    }

    # modify patient dataframe to create unique identifiers for each patient
    patient_df = pd.read_csv(LOCAL_PATIENT_TAB_PATH)
    patient_df['unique_id'] = patient_df['Patient'] + '___' + patient_df['Weeks'].astype(str)
    patient_keys = patient_df['unique_id'].unique().tolist() # like ID00007637202177411956430___7

    # # get list of all images
    # images_list = [os.path.splitext(f)[0] for f in os.listdir(LOCAL_PATIENT_MASKS_DIR)]
    # patients_ids = list(set([p.split('_')[0] for p in images_list])) # unique patients, not used rn

    # get partition and labels dicts for datagenerator class
    partition = {} # {'train':[<ids>], 'validation':[<ids>]}
    labels = {}    # {<id>:y_true, ...}

    # Assign train/validation keys
    random.shuffle(patient_keys)
    split_ind = int(np.floor(len(patient_keys)*validation_split))
    partition['train'] = patient_keys[:split_ind]
    partition['validation'] = patient_keys[split_ind:]
    # assign labels
    for kv in patient_df[['unique_id','FVC']].values: # [key, val]
        labels[kv[0]] = kv[1]

    # create internal dictionary for DataGenerators to use for processed tabular data
    tab_data = {} # {<id>: np.array(<encoded tabular data>)}
    no_op_attrs, num_attrs, cat_attrs, encoded_attrs = get_pipeline_selectors()
    pipeline = get_postproc_pipeline(
        no_op_attrs, num_attrs, cat_attrs, encoded_attrs)
    X = pipeline.fit_transform(patient_df).toarray()
    for unique_id, arr in zip(patient_df['unique_id'].values, X): # [key, val]
        tab_data[unique_id] = arr

    training_generator = myDataGenerator(list_ids=partition['train'],
        labels=labels,
        tab_data=tab_data,
        **datagen_params)
    validation_generator = myDataGenerator(list_ids=partition['validation'],
        labels=labels,
        tab_data=tab_data,
        **datagen_params)
    
    return training_generator, validation_generator


def main():
    LOCAL_PATIENT_TAB_PATH = "./data/train.csv"
    LOCAL_PATIENT_MASKS_DIR = "./data/processed_data/patient_masks_224/"

    # Load masked images into datagenerators
    training_generator, validation_generator = load_training_dataset(
        LOCAL_PATIENT_MASKS_DIR=LOCAL_PATIENT_MASKS_DIR,
        LOCAL_PATIENT_TAB_PATH=LOCAL_PATIENT_TAB_PATH
    )

    # lets just try predicting FVC from images alone
    model = build_wide_and_deep()
    model.summary()

    # train model
    model.fit(
        x=training_generator,
        epochs=2,
        verbose=1,
        validation_data=validation_generator
        )

if __name__ == '__main__':
    main()
    