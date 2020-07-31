import os
import shutil
import pickle
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import UpSampling3D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

############################################################
##################### helper functions #####################
############################################################
def normalize(data):
    data_min = data.min()
    data_max = data.max()
    data = [(x-data_min) / (data_max - data_min) for x in data]
    return np.array(data)

def merge_dicts(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 

############################################################
############### first let's define the model ###############
############################################################
def create_experimental_autoencoder(img_px_size=64, slice_count=8):
    """
    this model assumes (64, 64, 8) is the dimensionality of the 3D input
    """
    tf.keras.backend.set_image_data_format('channels_last')

    IMG_PX_SIZE = img_px_size
    SLICE_COUNT = slice_count

    input_shape = (IMG_PX_SIZE, IMG_PX_SIZE, SLICE_COUNT, 1)
    input_img = Input(shape=input_shape)

    # encoder portion
    x = Conv3D(50, (5, 5, 5), activation='relu', padding="same")(input_img)
    x = MaxPooling3D((2, 2, 2), padding="same")(x)
    x = Conv3D(50, (3, 3, 3), activation='relu', padding="same")(x)
    x = MaxPooling3D((2, 2, 2), padding="same")(x)
    x = Conv3D(50, (3, 3, 3), activation='relu', padding="same")(x)
    x = MaxPooling3D((2, 2, 2), padding="same")(x)
    x = Flatten()(x)
    encoded = Dense(500, activation="relu")(x)
    # at this point the representation is compressed to 500 dims

    # decoder portion
    x = Dense(3200, activation="relu")(encoded)
    x = Reshape((8, 8, 1, 50))(x)
    x = Conv3D(50, (3, 3, 3), activation='relu', padding="same")(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(50, (3, 3, 3), activation='relu', padding="same")(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(50, (5, 5, 5), activation='relu', padding="same")(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding="same")(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    encoder = Model(input_img, encoded)

    autoencoder.summary()

    return autoencoder, encoder

def create_autoencoder(img_px_size=32, slice_count=8):
    """
    this model assumes (32, 32, 8) is the dimensionality of the 3D input
    """
    tf.keras.backend.set_image_data_format('channels_last')

    IMG_PX_SIZE = img_px_size
    SLICE_COUNT = slice_count

    input_shape = (IMG_PX_SIZE, IMG_PX_SIZE, SLICE_COUNT, 1)
    input_img = Input(shape=input_shape)

    # encoder portion
    x = Conv3D(16, (3, 3, 3), activation='relu', padding="same")(input_img)
    x = MaxPooling3D((2, 2, 2), padding="same")(x)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding="same")(x)
    x = MaxPooling3D((2, 2, 2), padding="same")(x)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding="same")(x)
    encoded = MaxPooling3D((2, 2, 2), padding="same")(x)
    # at this point the representation is compressed to 4*4*8 = 128 dims
    # decoder portion
    x = Conv3D(8, (3, 3, 3), activation='relu', padding="same")(encoded)
    x = UpSampling3D((2, 2, 2))(encoded)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding="same")(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(16, (3, 3, 3), activation='relu', padding="same")(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding="same")(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer = 'adam', loss='binary_crossentropy')
    encoder = Model(input_img, encoded)

    return autoencoder,encoder

###############################################################
############### now lets grab the training data ###############
###############################################################
def get_training__and_validation_data_and_patients(filename="./data/processed_data/allthedata-condensed-with_ids-32-32-8.npy"):
    all_data = np.load(filename, allow_pickle=True)
    np.random.shuffle(all_data)

    train_data = all_data[:-50]
    imgs_x = train_data[:, 0]
    imgs_x = np.array([x[:] for x in imgs_x])
    train_x = np.moveaxis(imgs_x, 1, -1)
    patient_ids_x = train_data[:, 1]
    normalized_training_images = normalize(train_x)

    val_data = all_data[-50:]
    imgs_validation = val_data[:, 0]
    imgs_validation = np.array([x[:] for x in imgs_validation])
    validation_x = np.moveaxis(imgs_validation, 1, -1)
    patient_ids_validation = val_data[:, 1]
    normalized_validation_images = normalize(validation_x)

    return normalized_training_images, normalized_validation_images, patient_ids_x, patient_ids_validation

###########################################
################ lets train ###############
###########################################
def train_model(model, training_data, val_data, save_model=False, suffix=None, n_epochs=10):
    print("Training data shape: {}".format(training_data.shape))
    print("Validation data shape: {}".format(val_data.shape))
    print('Min: %.3f, Max: %.3f' % (training_data.min(), training_data.max()))
    print('Min: %.3f, Max: %.3f' % (val_data.min(), val_data.max()))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='binary_crossentropy')

    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'./autoencoder_model_{"" if suffix is None else suffix}_{datetime.datetime.now()}',
        monitor='val_loss',
        save_best_only=True
    )
    tensorboard_callback = TensorBoard(
        log_dir='/tmp/autoencoder'
    )

    model.fit(training_data, training_data,
                    epochs=n_epochs,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(val_data, val_data),
                    callbacks=[tensorboard_callback, model_checkpoint_callback])
    # if save_model:
    #     if not suffix:
    #         suffix = datetime.datetime.now()
    #     else:
    #         suffix += "-{}".format(datetime.datetime.now())
    #     model.save('./autoencoder_model_{}'.format(datetime.datetime.now()))

##############################################################
################ function for encoding patients ##############
##############################################################
def encode_patients(patient_ids, patient_images, model):
    patient_to_encoding_dict = {}
    
    for (img, patient_id) in zip(patient_images, patient_ids):
        encoding = model.predict(np.array([img])).flatten()
        patient_to_encoding_dict[patient_id] = encoding

    return patient_to_encoding_dict

def main():
    encoding_filepath = './data/processed_data/patient_ids_to_encodings_dict'
    # removing the directory for TensorBoard logs if it already exists
    if os.path.exists('/tmp/autoencoder'):
        shutil.rmtree('/tmp/autoencoder')

    autoencoder, encoder = create_experimental_autoencoder()
    training_data, val_data, training_patients, val_patients = get_training__and_validation_data_and_patients("./data/processed_data/171-images-with_ids-64-64-8.npy")
    train_model(autoencoder, training_data, val_data, n_epochs=60, save_model=True)
    encoded_training_patients = encode_patients(training_patients, training_data, encoder)
    encoded_val_patients = encode_patients(val_patients, val_data, encoder)
    
    all_encoded_patients = merge_dicts(encoded_training_patients, encoded_val_patients)
    
    encoding_filepath += "-{}".format(datetime.datetime.now())
    print("{} Patients encoded.".format(len(all_encoded_patients)))
    print("Saving to {}.pkl".format(encoding_filepath))

    with open(encoding_filepath + '.pkl', 'wb') as f:
        pickle.dump(all_encoded_patients, f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()