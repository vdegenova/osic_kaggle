import pydicom          # for reading dicom files
import os               # do directory ops
import pandas as pd     # data analysis
import numpy as np      # array ops
import math             # used for ceil ops
from cv2 import resize  # image processing
import matplotlib.pyplot as plt

def chunks(lst, n):
    # source link: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# I define this myself because I use it in map(), and pasing in an argument for which axis to average over to numpy.mean wouldn't work with map()
def mean(l):
    return sum(l)/len(l)


def hu_scaled_px(img, modality, rescale_slope, rescale_intercept):
    '''
    rescales array based on input parameters to fit HU scale (https://en.wikipedia.org/wiki/Hounsfield_scale)
    Air: -1000
    Water: 0
    Bone: 300-3000
    Metal: 2000
    INPUTS:
        img (np.array) array to be rescaled (dicom.pixel_array)
        modality (str): dicom attribute, modality
        rescale_slope (float): rescale slope (dicom.RescaleSlope)
        rescale_intercept (float): rescale slope (dicom.RescaleIntercept)
    RETRUNS:
        (np.array) scaled array
    '''
    return img if modality == "CR" else img * rescale_slope + rescale_intercept


def process_data(patient, patient_history_df, img_px_size=32, hm_slices=8, visualize=False, data_dir="./data/train/"):
    '''
    Function to read in all of the DICOMS in a patient dir, condense arrays into aggregated chunks
    INPUTS:
        patient (str): Patient ID. data_dir/<patient>/ contains DICOMS
        patient_history_df (pd.DataFrame): dataframe of tabular data associated with this patient
        img_px_size (int): resize source DICOM image array to square matrix of this shape
        hm_slices (int): number of chunks to output
    RETURNS:
        img_data (np.Array): (hm_slices, img_px_size, img_px_size) array for one patient
        patient_history (pd.DataFrame): Patient tabular data, filtered to critical columns
    '''

    path = data_dir + patient
    dicoms = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    dicoms.sort(key = lambda x: int(x.ImagePositionPatient[2])) # sorts DICOMS by caudial (ass) to cranial (head)
    
    # each scan is not the same depth (number of slices), we we group the slices into chunks of size HM_SLICES
    # and average across them to make sure the dimensionality is standardized
    
    new_slices = []

    # get pixel arrays for each dicom, HU rescale at the same time
    slices = [hu_scaled_px(dicom.pixel_array,
                           dicom.Modality,
                           dicom.RescaleSlope,
                           dicom.RescaleIntercept) for dicom in dicoms]
    
    # resize each pixel array - slices changed type to array here. Start as 512 x 512
    slices = [resize(np.array(each_slice), (img_px_size, img_px_size)) for each_slice in slices]

    # try opencv.resize on a different axis
    slices = np.array(slices)
    # initialze resized array, then modify each slice along second axis
    chunked_slices = np.zeros(shape=(hm_slices, img_px_size, img_px_size)) # (8, 64, 64)
    for i in range(chunked_slices.shape[1]):
        chunked_slices[:,i,:] = resize(slices[:,i,:], (img_px_size, hm_slices))
        # interpolation methods are: (https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/)
        # INTER_NEAREST – a nearest-neighbor interpolation
        # INTER_LINEAR – a bilinear interpolation (used by default)
        # INTER_AREA – resampling using pixel area relation.
        #   It may be a preferred method for image decimation, as it gives moire’-free results.
        #   But when the image is zoomed, it is similar to the INTER_NEAREST method. 
        # INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
        # INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood

    relevant_side_info = patient_history_df[["Patient", "Weeks", "FVC", "Percent"]]
    
    return chunked_slices, relevant_side_info

def read_in_data(data_dir="./data/train/", img_px_size=32, slice_count=8):
    '''
    Function to ___
    INPUTS:
        data_dir (str): folder location of training DICOMS patient folders
        img_px_size (int): resize source DICOM image array to square matrix of this shape
        hm_slices (int): number of chunks to condense each patient DICOMs into
    RETURNS:
        all_the_data (np.array): array of size (n_patients, 2), where each entry is [DICOM_reshape_array, patient_id]
    '''

    patients = os.listdir(data_dir) # list of training patient IDS (folders which contain DICOMS)
    train_df = pd.read_csv("./data/train.csv") # list of training patient tabular data

    IMG_PX_SIZE = img_px_size
    SLICE_COUNT = slice_count

    error_log = []
    all_the_data = []

    for num, patient in enumerate(patients):
        if num%10 == 0:
            print("Patient:" + str(num))
        patient_history_df = train_df[train_df.Patient == patient].sort_values(by="Weeks")
        
        try:
            img_data, patient_history = process_data(patient, patient_history_df, img_px_size=IMG_PX_SIZE, hm_slices=SLICE_COUNT)
            patient_id = patient_history.Patient.iloc[0]
            all_the_data.append([img_data, patient_id])
                        
        except Exception as e:
            print(patient, e)
            error_log.append((patient, e))
            continue
    
    return np.array(all_the_data, dtype=object)

def save_to_disk(data, img_px_size=32, slice_count=8):
    filestring = f'./data/processed_data/{data.shape[0]}-images-with_ids-{img_px_size}-{img_px_size}-{slice_count}-{datetime.datetime.now()}.npy'
    print(f'saving to {filestring}')
    np.save(filestring, data)

def main():
    patient_data = read_in_data(data_dir="./data/train/", img_px_size=64, slice_count=8)
    save_to_disk(patient_data, img_px_size=64, slice_count=8)

if __name__ == "__main__":
    main()