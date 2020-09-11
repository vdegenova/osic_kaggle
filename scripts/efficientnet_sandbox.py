# playing with efficientnet
# install efficientnet with `pip install -U efficientnet` :)
import numpy as np
import json
from typing import Tuple
import efficientnet.tfkeras as efn
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate
)

# https://kobiso.github.io/Computer-Vision-Leaderboard/imagenet.html
# B0 expects 224x224
# B1 expects 240x240
# B2 expects 260x260
# B3 expects 300x300
# B4 expects 380x380
# B5 expects 456x456
# B6 expects 528x528
# B6 expects 600x600

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
    inp = Input(shape=input_shape)
    base = efn.EfficientNetB0(input_shape=input_shape, weights=weights, include_top=False)
    for layer in base.layers:
        layer.trainable = False
    
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1)(x)

    model = Model(inp, x)
    return model


def load_masked_image_data(PATIENT_IMAGES_FILEPATH:str) -> dict:
    print(f'reading {PATIENT_IMAGES_FILEPATH}')
    with open(PATIENT_IMAGES_FILEPATH, 'r') as f:
        data = json.load(f)
    # anticipated structure is data[<patient_id>]=<nested list of image arrays>
    # convert image arrays back into numpy
    for key, val in data.items():
        data[key] = np.array(val)
    return data


def load_training_dataset():
    pass

def main():
    LOCAL_PATIENT_DIR = "./data/train/"
    LOCAL_PATIENT_IMAGES_FILEPATH = "./data/processed_data/3-images-with_ids-224-224-None-2020-09-11T11:44.json"

    # Load masked images
    data = load_masked_image_data(PATIENT_IMAGES_FILEPATH=LOCAL_PATIENT_IMAGES_FILEPATH)
    print(data['ID00196637202246668775836'].shape)

    # lets just try predicting FVC from images alone
    model = build_wide_and_deep()
    model.summary()



if __name__ == '__main__':
    main()
    