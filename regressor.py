import os
import shutil
import pickle
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split

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
def create_dense_regressor(n_input_dims):
    """
    Creates and returns model architecture for a dense network with dropout
    Input(520)  --> Dense(1024) --> droput(0.5) --> Dense(512) --> dropout(0.5) -->Dense(256) -->dropout(0.5) -->Dense(1)
    """

    input_shape = n_input_dims
    input_features = Input(shape=input_shape)

    # model arch
    x = Dense(1024, activation='relu',)(input_features)
    x = Dropout(.5)(x)
    x = Dense(512, activation='relu',)(x)
    x = Dropout(.5)(x)
    x = Dense(256, activation='relu',)(x)
    x = Dropout(.5)(x)
    fvc_prediction = Dense(1)(x)

    regressor = Model(input_features, fvc_prediction)
    regressor.compile(optimizer='adam', loss='binary_crossentropy')

    regressor.summary()

    return regressor


###########################################
################ lets train ###############
###########################################
def train_model(model, training_data, val_data, suffix=None, n_epochs=10):
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

    # removing the directory for TensorBoard logs if it already exists
    if os.path.exists('/tmp/regressor'):
        shutil.rmtree('/tmp/regressor')

    # pass embedding + csv filepaths to function to unify into one dataframe
    embeddings_path = './data/processed_data/patient_ids_to_encodings_dict-2020-07-31 17:09:25.371193.pkl'
    tabular_path = './data/train_csv'
    all_data = ...

    # get, run pipeline
    pipeline = ...
    X, y = pipeline.fit_transform(all_data)

    # test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=49)

    # Create regressor
    regressor = create_dense_regressor(n_input_dims = X_train.shape[1])

    # train model
    train_model(regressor, training_data, val_data, n_epochs=120)

    # encoded_training_patients = encode_patients(training_patients, training_data, encoder)
    # encoded_val_patients = encode_patients(val_patients, val_data, encoder)
    # all_encoded_patients = merge_dicts(encoded_training_patients, encoded_val_patients)
    # encoding_filepath += "-{}".format(datetime.datetime.now())
    # print("{} Patients encoded.".format(len(all_encoded_patients)))
    # print("Saving to {}.pkl".format(encoding_filepath))

    # with open(encoding_filepath + '.pkl', 'wb') as f:
    #     pickle.dump(all_encoded_patients, f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()