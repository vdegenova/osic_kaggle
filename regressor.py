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

# custom modules
from post_proc import get_pipeline_selectors, get_postproc_pipeline, load_pickled_encodings

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
def tilted_loss(y, f):
    q = 0.9 # which quantile are we predicting (0.9 = 90th percentile)
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

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
    x = Dense(512, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(.5)(x)
    fvc_prediction = Dense(1)(x)

    regressor = Model(input_features, fvc_prediction)
    quantile = 0.9
    regressor.compile(loss=tilted_loss, optimizer='adam')
    # regressor.compile(optimizer='adam', loss='LogCosh')

    regressor.summary()

    return regressor


###########################################
################ lets train ###############
###########################################
def train_model(model, X, y, validation_split, suffix=None, n_epochs=10):
    # print("Training data shape: {}".format(training_data.shape))
    # print("Validation data shape: {}".format(val_data.shape))
    # print('Min: %.3f, Max: %.3f' % (training_data.min(), training_data.max()))
    # print('Min: %.3f, Max: %.3f' % (val_data.min(), val_data.max()))

    # model was already compiled with Adam in create_()????
    # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # model.compile(optimizer=opt, loss='binary_crossentropy')

    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'./regressor_model_{"" if suffix is None else suffix}_{datetime.datetime.now()}',
        monitor='val_loss',
        save_best_only=True
    )
    tensorboard_callback = TensorBoard(
        log_dir='/tmp/regressor'
    )

    model.fit(X, y,
        epochs=n_epochs,
        batch_size=32,
        shuffle=True,
        validation_split=validation_split,
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
        print("Removing existing tensorboard logs")
        shutil.rmtree('/tmp/regressor')

    # pass embedding + csv filepaths to function to unify into one dataframe
    #embeddings_path = './data/processed_data/patient_ids_to_encodings_dict-2020-07-31 17:09:25.371193.pkl'
    embeddings_path = './data/processed_data/patient_ids_to_encodings_dict-2020-07-26 13:50:14.433337.pkl'
    csv_path = './data/train.csv'
    all_data = load_pickled_encodings(embeddings_path, csv_path)
    X = all_data.drop(columns = 'FVC') # keep as a dataframe to pass to pipeline
    y = all_data[['FVC']]

    # get, run pipeline
    no_op_attrs, num_attrs, cat_attrs, encoded_attrs = get_pipeline_selectors()
    pipeline = get_postproc_pipeline(no_op_attrs, num_attrs, cat_attrs, encoded_attrs)
    X = pipeline.fit_transform(X).toarray() # returns scipy.sparse.csr.csr_matrix by default

    # Create regressor
    regressor = create_dense_regressor(n_input_dims = X.shape[1])

    # train model
    train_model(regressor, X, y, validation_split=.3, n_epochs=100, suffix='quantile_90')

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