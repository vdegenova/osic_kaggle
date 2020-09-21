# playing with efficientnet
# install efficientnet with `pip install -U efficientnet` :)
import numpy as np
import json
import os
import random
import datetime
import pandas as pd
from typing import Tuple
import efficientnet.tfkeras as efn
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D,
    LeakyReLU, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from pipelines import (
    get_pipeline_selectors,
    get_postproc_pipeline,
    load_pickled_encodings,)
from myDataGenerator import myDataGenerator
import sys
sys.path.append('./src/')

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
    img_input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights='noisy-student',
    tab_input_shape: Tuple[int, ] = (7,),
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

    # Create Deep leg
    inp_img = Input(shape=img_input_shape)
    base = efn.EfficientNetB0(
        input_shape=img_input_shape, weights=weights, include_top=False)
    for layer in base.layers:
        layer.trainable = False
    x = base(inp_img)
    x = GlobalAveragePooling2D()(x)

    # Create Wide leg
    inp_tab = Input(shape=tab_input_shape)
    x = Concatenate()([x, inp_tab])

    x = Dense(1)(x)  # activation defaults to linear

    model = Model([inp_tab, inp_img], x)

    # compile model
    opt = Adam(learning_rate=1e-5)
    loss = 'MSE'
    model.compile(optimizer=opt, loss=loss)

    return model


def load_training_dataset(LOCAL_PATIENT_MASKS_DIR: str, LOCAL_PATIENT_TAB_PATH: str,
                          validation_split=0.7):
    # converts a loaded data dict of masked images into a proper dataset for NN training
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    datagen_params = {
        'dim': (224, 224),
        'batch_size': 32,
        'n_channels': 3,
        'data_dir': LOCAL_PATIENT_MASKS_DIR
    }

    # modify patient dataframe to create unique identifiers for each patient
    patient_df = pd.read_csv(LOCAL_PATIENT_TAB_PATH)
    patient_df['unique_id'] = patient_df['Patient'] + \
        '___' + patient_df['Weeks'].astype(str)

    # get list of all images - need in order to remove patients that could not be masked
    images_list = [os.path.splitext(f)[0]
                   for f in os.listdir(LOCAL_PATIENT_MASKS_DIR)]
    # unique patients, not used rn
    patients_with_masks = list(set([p.split('_')[0] for p in images_list]))
    patient_keys = patient_df[patient_df['Patient'].isin(
        patients_with_masks)]['unique_id'].unique().tolist()  # like ID00007637202177411956430___7

    # run FVC through standardization pipeline
    fvc_pipeline = get_postproc_pipeline(num_attrs=['FVC'])
    std_fvc = fvc_pipeline.fit_transform(patient_df)  # .toarray()
    patient_df['FVC'] = std_fvc

    # get partition and labels dicts for datagenerator class
    partition = {}  # {'train':[<ids>], 'validation':[<ids>]}
    labels = {}    # {<id>:y_true, ...}

    # Assign train/validation keys
    random.shuffle(patient_keys)
    split_ind = int(np.floor(len(patient_keys)*validation_split))
    partition['train'] = patient_keys[:split_ind]
    partition['validation'] = patient_keys[split_ind:]
    # assign labels
    for kv in patient_df[['unique_id', 'FVC']].values:  # [key, val]
        labels[kv[0]] = kv[1]

    # create internal dictionary for DataGenerators to use for processed tabular data
    tab_data = {}  # {<id>: np.array(<encoded tabular data>)}
    no_op_attrs, num_attrs, cat_attrs, encoded_attrs = get_pipeline_selectors()
    pipeline = get_postproc_pipeline(
        no_op_attrs, num_attrs, cat_attrs, encoded_attrs)
    X = pipeline.fit_transform(patient_df).toarray()
    for unique_id, arr in zip(patient_df['unique_id'].values, X):  # [key, val]
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


def train_model(model, training_generator, validation_generator, n_epochs=10, suffix=None):
    '''
    ::param model:: model to train
    ::param training_generator:: training batch generator - feeds image and tabular data
    ::param validation_generator:: validation batch generator
    ::param n_epochs:: number of epochs to train
    ::param suffix:: custom string to add to model
    '''

    # prepare model checkpoint callback
    now = datetime.datetime.now().isoformat(timespec="minutes")
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'./models/wide_and_deep_model_{"" if suffix is None else suffix}_{now}',
        monitor="val_loss",
        save_best_only=True,
    )
    # prepare tensorboard callback
    log_dir = "/tmp/wide_and_deep/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=10
    )  # call tensorboard with `tensorboard --logdir /tmp/wide_and_deep` from your env

    # train
    model.fit(
        x=training_generator,
        epochs=n_epochs,
        verbose=1,
        validation_data=validation_generator,
        callbacks=[tensorboard_callback, model_checkpoint_callback]
    )


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
    train_model(model=model,
                training_generator=training_generator,
                validation_generator=validation_generator,
                n_epochs=1000
                )


if __name__ == '__main__':
    main()
