from tensorflow import keras
import numpy as np
import pandas as pd
import random
import os


class myDataGenerator(keras.utils.Sequence):
    '''
    Generates data for keras
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    ATTRS:
        list_ids (list): list of ids to include. This is how we separate the training and validation dataloaders
        labels (dict): dictionary mapping of {id:y_true}
        data_dir (str): path to where our image data is found
        tab_data (dict): internal resource mapping {id:np.array(tabular data, post pipeline)}
        batch_size (int): how many data points to generate per call
        dim (tuple): tuple of ints for image data specifying size and channels. honest to god dont know why we need this and n_channels
        n_channels (int): if channels is > 1, it will stack the greyscale images into multiple channels
        tab_data_dim (int): how many elements to expect in the tabular data array
        n_classes (int): not used for regression
    '''
    def __init__(self, list_ids, labels, data_dir, tab_data, batch_size=32, dim=(224,224,3), n_channels=1, tab_data_dim=7,
                 n_classes=None, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_dir = data_dir
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tab_data = tab_data
        self.tab_data_dim = tab_data_dim
        self.on_epoch_end()

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
        x_imgs = np.empty((self.batch_size, *self.dim, self.n_channels))
        x_tab = np.empty((self.batch_size, self.tab_data_dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # remember ID here is PatientID___Weeks
            patient_id_no_weeks = ID.split('___')[0]
            # get image data
            all_patient_images = [f for f in os.listdir(self.data_dir) if patient_id_no_weeks in f]
            #greyscale_img = np.load(self.data_dir + patient_id_no_weeks + '.npy')
            greyscale_img = np.load(self.data_dir +random.choice(all_patient_images))
            try:
                assert not np.any(np.isnan(greyscale_img))
            except AssertionError as e:
                e.args += ('###############', ID, '###############')
                raise
            
            # get tabular data
            x_tab[i,] = self.tab_data[ID]

            if self.n_channels > 1:
                x_imgs[i,] = np.stack((greyscale_img,)*self.n_channels, axis=-1) # here I am making 1 channel into x duplicate channels
            else:
                x_imgs[i,] = greyscale_img

            # Store class
            y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes) # wont need for regression
        return [x_tab, x_imgs], y