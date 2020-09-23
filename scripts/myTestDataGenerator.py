from tensorflow import keras
import numpy as np
import pandas as pd
import random
import os


class myTestDataGenerator(keras.utils.Sequence):
    '''
    Generates data for keras
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Each batch should be 1 patient, 1 week, but all dicoms
    ATTRS:
        data_dir (str): path to where our image data is found
        tab_data (dict): internal resource mapping {id:np.array(tabular data, post pipeline)}
        dim (tuple): tuple of ints for image data specifying size and channels. honest to god dont know why we need this and n_channels
        n_channels (int): if channels is > 1, it will stack the greyscale images into multiple channels
        tab_data_dim (int): how many elements to expect in the tabular data array
        n_classes (int): not used for regression
        patient_slices_library (dict): {<patient_id>:<list of sorted np arrays>} used to keep arrays in-memory

        df (pd.DataFrame): the testing df which includes 'Patient' as well as unpipelined patient data
        tab_pipeline (sklearn.pipeline.Pipeline): a trained pipeline to run the test tab data through
    '''
    def __init__(self, df, tab_pipeline, data_dir, tab_data, dim=(224,224,3), n_channels=1,
                 tab_data_dim=7, patient_slices_library={}, n_classes=None, shuffle=False):
        self.dim = dim
        self.batch_size = 1
        self.data_dir = data_dir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tab_data_dim = tab_data_dim
        self.on_epoch_end()
        self.tab_pipeline = tab_pipeline
        self.df = df
        self.patient_slices_library = patient_slices_library
        self.list_ids = df['unique_id'].unique()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_ids) / self.batch_size))
    

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y


    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_ids_temp):
        '''Generates data containing batch_size samples''' # X : (n_samples, *dim, n_channels)
        # each item of X needs to be a tuple. The first item can be the image, the second must be tabular
        # Initialization

        # calculate the pseudo batch-size: is number of dicoms
        pseudo_batch = len(self.patient_slices_library[patient_id_no_weeks]))
        x_imgs = np.empty((pseudo_batch, *self.dim, self.n_channels))
        x_tab = np.empty((pseudo_batch, self.tab_data_dim))

        # Generate data
        ID = list_ids_temp[0]
        # remember ID here is PatientID___Weeks
        patient_id_no_weeks = ID.split('___')[0]

        # get tabular data
        x_tab = np.stack((self.tab_data[ID],)*pseudo_batch, axis=0)
        
        # get image data
        # read in-memory
        for i, greyscale_img in enumerate(self.patient_slices_library[patient_id_no_weeks])
            try:
                assert not np.any(np.isnan(greyscale_img))
            except AssertionError as e:
                e.args += ('###############', ID, '###############')
                raise

            if self.n_channels > 1:
                x_imgs[i,] = np.stack((greyscale_img,)*self.n_channels, axis=-1) # here I am making 1 channel into x duplicate channels
            else:
                x_imgs[i,] = greyscale_img

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes) # wont need for regression
        return [x_tab, x_imgs]