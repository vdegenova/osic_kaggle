import os
import shutil
import pickle
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import UpSampling3D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from volume_image_gen import customImageDataGenerator
import kerastuner as kt

############################################################
##################### helper functions #####################
############################################################
def normalize(data):
    # TODO: swap with RobustScaler
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

    initializer = tf.keras.initializers.GlorotNormal()


    # encoder portion
    x = Conv3D(20, (5, 5, 5), activation='relu', padding="same", kernel_initializer=initializer)(input_img)
    x = MaxPooling3D((2, 2, 2), padding="same")(x)
    x = Conv3D(20, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer)(x)
    x = MaxPooling3D((2, 2, 2), padding="same")(x)
    x = Conv3D(20, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer)(x)
    x = MaxPooling3D((2, 2, 2), padding="same")(x)
    x = Flatten()(x)
    encoded = Dense(500, activation="relu", kernel_initializer=initializer)(x)
    # at this point the representation is compressed to 500 dims

    # decoder portion
    x = Dense(3200, activation="relu", kernel_initializer=initializer)(encoded)
    x = Reshape((8, 8, 1, 50))(x)
    x = Conv3D(20, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer)(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(20, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer)(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(20, (5, 5, 5), activation='relu', padding="same", kernel_initializer=initializer)(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding="same", kernel_initializer=initializer)(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

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
    # autoencoder.compile(optimizer = 'adam', loss='binary_crossentropy')
    encoder = Model(input_img, encoded)

    return autoencoder, encoder


def create_jesse_autoencoder(img_px_size=64, slice_count=8):
    """
    this model assumes (64, 64, 8) is the dimensionality of the 3D input
    """
    tf.keras.backend.set_image_data_format('channels_last')

    IMG_PX_SIZE = img_px_size
    SLICE_COUNT = slice_count
    input_shape = (IMG_PX_SIZE, IMG_PX_SIZE, SLICE_COUNT, 1)
    initializer = tf.keras.initializers.GlorotNormal()

    # model architecture hyperparams
    conv1_kernels = 30
    conv2_kernels = 20
    conv3_kernels = 30
    learning_rate = 0.0015

    # encoder portion
    encoder = keras.Sequential(
        [
            Conv3D(conv1_kernels, (5, 5, 5), activation='relu', padding="same", input_shape=input_shape, kernel_initializer=initializer),
            MaxPooling3D((2, 2, 2), padding="same"),
            Conv3D(conv2_kernels, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer),
            MaxPooling3D((2, 2, 2), padding="same"),
            Conv3D(conv3_kernels, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer),
            MaxPooling3D((2, 2, 2), padding="same"),
            Flatten(),
            Dense(500, activation="relu", kernel_initializer=initializer)
        ], name = 'sequential_encoder'
    ) # at this point the representation is compressed to 500 dims

    # decoder portion
    decoder = keras.Sequential(
        [
            Dense(3200, activation="relu", input_shape=(500,), kernel_initializer=initializer),
            Reshape((8, 8, 1, 50)),
            Conv3D(conv3_kernels, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer),
            UpSampling3D((2, 2, 2)),
            Conv3D(conv2_kernels, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer),
            UpSampling3D((2, 2, 2)),
            Conv3D(conv1_kernels, (5, 5, 5), activation='relu', padding="same", kernel_initializer=initializer),
            UpSampling3D((2, 2, 2)),
            Conv3D(1, (3, 3, 3), activation='sigmoid', padding="same", kernel_initializer=initializer)
        ], name = 'sequential_decoder'
    )

    # autoencoder sequential model
    autoencoder = keras.Sequential(
        [
            encoder,
            decoder
        ], name = 'sequential_autoencoder'
    )

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # opt = tf.keras.optimizers.Adadelta(learning_rate=1e-5)
    loss = 'binary_crossentropy'
    # loss = 'MSE'
    autoencoder.compile(optimizer=opt, loss=loss)
    autoencoder.summary()

    return autoencoder, encoder

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
def train_model(model, training_data, val_data, suffix=None, n_epochs=10):
    print("Training data shape: {}".format(training_data.shape))
    print("Validation data shape: {}".format(val_data.shape))
    print('Min: %.3f, Max: %.3f' % (training_data.min(), training_data.max()))
    print('Min: %.3f, Max: %.3f' % (val_data.min(), val_data.max()))

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='logcosh')
    # model.summary()

    # prepare model checkpoint callback
    now = datetime.datetime.now().isoformat(timespec='minutes')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'./models/autoencoder_model_{"" if suffix is None else suffix}_{now}',
        monitor='val_loss',
        save_best_only=True
    )
    # prepare tensorboard callback
    log_dir='/tmp/autoencoder/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=3
        ) # call tensorboard with tensorboard --logdir /tmp/autoencoder

    # train model
    model.fit(training_data, training_data,
                    epochs=n_epochs,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(val_data, val_data),
                    callbacks=[tensorboard_callback, model_checkpoint_callback])


def train_with_augmentation(model, training_data, val_data, suffix=None, n_epochs=10):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    '''
    print("Training data shape: {}".format(training_data.shape))
    print("Validation data shape: {}".format(val_data.shape))
    print('Min: %.3f, Max: %.3f' % (training_data.min(), training_data.max()))
    print('Min: %.3f, Max: %.3f' % (val_data.min(), val_data.max()))

    # prepare model checkpoint callback
    now = datetime.datetime.now().isoformat(timespec='minutes')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'./models/autoencoder_model_{"" if suffix is None else suffix}_{now}',
        monitor='val_loss',
        save_best_only=True
    )
    # prepare tensorboard callback
    log_dir='/tmp/autoencoder/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=3
        ) # call tensorboard with tensorboard --logdir /tmp/autoencoder

    # prepare datagenerator(s)
    data_gen_args = dict(
        rotation_range=5,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True)

    input_datagen = customImageDataGenerator(**data_gen_args)
    output_datagen = customImageDataGenerator(**data_gen_args)

    # if using customImageDataGenerator, must exapand dims of training data
    if model.name == 'sequential_autoencoder':
        training_data = np.expand_dims(training_data, -1)
        val_data = np.expand_dims(val_data, -1)
    seed = 1
    input_generator = input_datagen.flow(training_data, batch_size=32, seed=seed)
    output_generator = output_datagen.flow(training_data, batch_size=32, seed=seed)

    train_generator = zip(input_generator, output_generator)

    # train model

    model.fit(train_generator,
        epochs=n_epochs,
        shuffle=True,
        steps_per_epoch=4,
        validation_data=(val_data, val_data),
        callbacks=[tensorboard_callback, model_checkpoint_callback])


##############################################################
################ function for encoding patients ##############
##############################################################
def encode_patients(patient_ids, patient_images, model):
    patient_to_encoding_dict = {}
    
    for (img, patient_id) in zip(patient_images, patient_ids):
        # will need to expand dims for sequential model to work, need rgb channel (see training with augementation)
        if model.name == 'sequential_encoder':
            img = np.expand_dims(img, -1)
        encoding = model.predict(np.array([img])).flatten()
        patient_to_encoding_dict[patient_id] = encoding

    return patient_to_encoding_dict


def kt_model_builder(hp, img_px_size=64, slice_count=8,):
    """
    This model definition is copied from create_jese_autoencoder
    KEY DIFFERENCE is I do only return the autoencoder, and not the encoder portion!
    this model assumes (64, 64, 8) is the dimensionality of the 3D input
    """

    # keras tuner! start with just a few convolution kernels as example
    hp_conv1_kernels = hp.Int('conv1_kernels', min_value=10, max_value=50, step=10)
    hp_conv2_kernels = hp.Int('conv2_kernels', min_value=10, max_value=50, step=10)
    hp_conv3_kernels = hp.Int('conv3_kernels', min_value=10, max_value=50, step=10)
    hp_lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)

    tf.keras.backend.set_image_data_format('channels_last')

    IMG_PX_SIZE = img_px_size
    SLICE_COUNT = slice_count
    input_shape = (IMG_PX_SIZE, IMG_PX_SIZE, SLICE_COUNT, 1)
    initializer = tf.keras.initializers.GlorotNormal()

    # encoder portion
    encoder = keras.Sequential(
        [
            Conv3D(hp_conv1_kernels, (5, 5, 5), activation='relu', padding="same", input_shape=input_shape, kernel_initializer=initializer),
            MaxPooling3D((2, 2, 2), padding="same"),
            Conv3D(hp_conv2_kernels, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer),
            MaxPooling3D((2, 2, 2), padding="same"),
            Conv3D(hp_conv3_kernels, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer),
            MaxPooling3D((2, 2, 2), padding="same"),
            Flatten(),
            Dense(500, activation="relu", kernel_initializer=initializer)
        ], name = 'sequential_encoder'
    ) # at this point the representation is compressed to 500 dims

    # decoder portion
    decoder = keras.Sequential(
        [
            Dense(3200, activation="relu", input_shape=(500,), kernel_initializer=initializer),
            Reshape((8, 8, 1, 50)),
            Conv3D(hp_conv3_kernels, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer),
            UpSampling3D((2, 2, 2)),
            Conv3D(hp_conv2_kernels, (3, 3, 3), activation='relu', padding="same", kernel_initializer=initializer),
            UpSampling3D((2, 2, 2)),
            Conv3D(hp_conv1_kernels, (5, 5, 5), activation='relu', padding="same", kernel_initializer=initializer),
            UpSampling3D((2, 2, 2)),
            Conv3D(1, (3, 3, 3), activation='sigmoid', padding="same", kernel_initializer=initializer)
        ], name = 'sequential_decoder'
    )

    # autoencoder sequential model
    autoencoder = keras.Sequential(
        [
            encoder,
            decoder
        ], name = 'sequential_autoencoder'
    )

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=hp_lr)
    # opt = tf.keras.optimizers.Adadelta(learning_rate=1e-5)
    loss = 'binary_crossentropy'
    # loss = 'MSE'
    autoencoder.compile(optimizer=opt, loss=loss)
    # autoencoder.summary()

    return autoencoder


def train_kt_model(training_data, val_data, n_epochs=10, suffix=None):
    '''
    This function is used to instantiate a keras tuner, and use to tune a hypermodel
    https://www.sicara.ai/blog/hyperparameter-tuning-keras-tuner
    '''
    # instantiate Hyperband tuner
    # https://arxiv.org/pdf/1603.06560.pdf
    tuner = kt.Hyperband(kt_model_builder,
        objective = 'val_loss', 
        max_epochs = 10,
        factor = 3,
        seed = 1,
        directory = '/tmp/kerastuner/',
        project_name = 'simple_autoencoder')

    # must clear the training outputs at the end of every training step... dunno why
    # class ClearTrainingOutput(tf.keras.callbacks.Callback):
    #     def on_train_end(*args, **kwargs):
    #         IPython.display.clear_output(wait = True)

    # prepare training generators
    print("Training data shape: {}".format(training_data.shape))
    print("Validation data shape: {}".format(val_data.shape))
    print('Min: %.3f, Max: %.3f' % (training_data.min(), training_data.max()))
    print('Min: %.3f, Max: %.3f' % (val_data.min(), val_data.max()))

    # prepare model checkpoint callback
    now = datetime.datetime.now().isoformat(timespec='minutes')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'./models/autoencoder_model_{"" if suffix is None else suffix}_{now}',
        monitor='val_loss',
        save_best_only=True
    )
    # prepare tensorboard callback
    log_dir = '/tmp/autoencoder/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=3
        ) # call tensorboard with tensorboard --logdir /tmp/autoencoder

    # prepare datagenerator(s)
    data_gen_args = dict(
        rotation_range=5,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True)

    input_datagen = customImageDataGenerator(**data_gen_args)
    output_datagen = customImageDataGenerator(**data_gen_args)

    # if using customImageDataGenerator, must exapand dims of training data
    training_data = np.expand_dims(training_data, -1)
    val_data = np.expand_dims(val_data, -1)
    seed = 1
    input_generator = input_datagen.flow(training_data, batch_size=32, seed=seed)
    output_generator = output_datagen.flow(training_data, batch_size=32, seed=seed)

    train_generator = zip(input_generator, output_generator)

    # train model
    tuner.search(train_generator,
        epochs=n_epochs,
        steps_per_epoch=4,
        validation_data=(val_data, val_data),
        )

    ############## evaluate best model ##############
    # Show a summary of the search
    tuner.results_summary()
    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]
    print("BEST MODEL SUMMARY:")
    try:
        for layer in best_model.layers:
            layer.summary()
    except:
        best_model.summary()
    # Evaluate the best model.
    loss = best_model.evaluate(val_data, val_data)
    print(f'FINAL RESULTS:\nloss: {loss}')


def main():
    ENCODING_FILEPATH = './data/processed_data/patient_ids_to_encodings_dict'
    PREPROCESSED_NPY = './data/processed_data/158-images-with_ids-64-64-8-2020-08-26T21:12.npy'
    KT_SESSION = False

    '''
    [Trial summary]
    |-Trial ID: 5cbb1ef77bdc8a39974d611520ebfd4a
    |-Score: 0.2765873968601227
    |-Best step: 0
    > Hyperparameters:
    |-conv1_kernels: 30
    |-conv2_kernels: 20
    |-conv3_kernels: 30
    |-learning_rate: 0.001565284883754403
    |-tuner/bracket: 0
    |-tuner/epochs: 10
    |-tuner/initial_epoch: 0
    |-tuner/round: 0
    '''

    # Load training + validation data from preprocessed .npy
    training_data, val_data, training_patients, val_patients = get_training__and_validation_data_and_patients(PREPROCESSED_NPY)
    
    if KT_SESSION:
        # train hypermodel
        train_kt_model(training_data, val_data, n_epochs=25)
    else:
        # Train regular model
        autoencoder, encoder = create_jesse_autoencoder()
        train_with_augmentation(autoencoder, training_data, val_data, n_epochs=100)

        # Record final embedding for each patient {PatientID: flatten(embeddings)}
        encoded_training_patients = encode_patients(training_patients, training_data, encoder)
        encoded_val_patients = encode_patients(val_patients, val_data, encoder)
        all_encoded_patients = merge_dicts(encoded_training_patients, encoded_val_patients)
        
        # Save embeddings as pkl of dictionary
        now = datetime.datetime.now().isoformat(timespec='minutes')
        ENCODING_FILEPATH += f"-{now}"
        print("{} Patients encoded.".format(len(all_encoded_patients)))
        print("Saving to {}.pkl".format(ENCODING_FILEPATH))

        with open(ENCODING_FILEPATH + '.pkl', 'wb') as f:
            pickle.dump(all_encoded_patients, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()