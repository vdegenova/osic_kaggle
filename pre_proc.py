import os                                   # do directory ops
import math                                 # used for ceil ops
import datetime                             # used for saving with a datetime string
import pandas as pd                         # data analysis
import numpy as np                          # array ops
import matplotlib.pyplot as plt             # used for visualization in dev
import pydicom                              # for reading dicom files
from cv2 import resize                      # image processing
from skimage import morphology, measure     # for lung masking
from sklearn.cluster import KMeans          # for lung masking
from tqdm import tqdm                       # for progress bars


def chunks(lst, n):
    # source link: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# I define this myself because I use it in map(), and pasing in an argument for which axis to average over to numpy.mean wouldn't work with map()
def mean(l):
    return sum(l)/len(l)


def hu_scaled_px(img, modality, rescale_slope, rescale_intercept, verbose=False):
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
    if verbose:
        print(f'\tRescaleSlope: {rescale_slope}, RescaleIntercept: {rescale_intercept}')

    if rescale_slope == 0 or rescale_slope is None or np.isnan(rescale_slope):
        return img
    else:
        return img if modality == "CR" else img * rescale_slope + rescale_intercept


# lungmask experimenting
def make_lungmask(img, display=False):
    '''
    - Standardize the pixel value by subtracting the mean and dividing by the standard deviation - does it tho?
    - Identify the proper threshold by creating 2 KMeans clusters comparing centered on soft tissue/bone vs lung/air.
    - Use Erosion and Dilation which has the net effect of removing tiny features like pulmonary vessels or noise
    - Identify each distinct region as separate image labels (think the magic wand in Photoshop)
    - Using bounding boxes for each image label to identify which ones represent lung and which ones represent "every thing else"
    - Create the masks for lung fields.
    - Apply mask onto the original image to erase voxels outside of the lung fields.
    INPUTS:
        img (np.array): pixel array containing a lung
        display (bool): plotting purposes only
    RETURN:
        (np.array): masked image pixel array
    '''
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs to renormalize washed out images.
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    # Use Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.
    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        # check if 20%_image < region < 90%_image for both width and height?
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[8, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img

def process_data(patient, patient_history_df, img_px_size=32, hm_slices=8, verbose=False, data_dir="./data/train/"):
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
    dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2])) # sorts DICOMS by caudial (ass) to cranial (head)

    # each scan is not the same depth (number of slices), we we group the slices into chunks of size HM_SLICES
    # and average across them to make sure the dimensionality is standardized
    # get pixel arrays for each dicom, HU rescale at the same time
    slices = [hu_scaled_px(dicom.pixel_array,
                           dicom.Modality,
                           dicom.RescaleSlope,
                           dicom.RescaleIntercept,
                           verbose=False) for dicom in dicoms]

    # EXPERIMENTAL: lung masking
    masked_slices = []
    for each_slice in tqdm(slices):
        masked_slices.append(make_lungmask(each_slice))

    # resize each pixel array - slices changed type to array here. Start as 512 x 512
    masked_slices = [resize(np.array(each_slice), (img_px_size, img_px_size)) for each_slice in slices]

    # try opencv.resize on a different axis
    masked_slices = np.array(slices)
    # initialze resized array, then modify each slice along second axis
    chunked_slices = np.zeros(shape=(hm_slices, img_px_size, img_px_size)) # (8, 64, 64)
    for i in range(chunked_slices.shape[1]):
        chunked_slices[:,i,:] = resize(masked_slices[:,i,:], (img_px_size, hm_slices))
        # interpolation methods are: (https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/)
        # INTER_NEAREST – a nearest-neighbor interpolation
        # INTER_LINEAR – a bilinear interpolation (used by default)
        # INTER_AREA – resampling using pixel area relation.
        #   It may be a preferred method for image decimation, as it gives moire’-free results.
        #   But when the image is zoomed, it is similar to the INTER_NEAREST method. 
        # INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
        # INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood

    relevant_side_info = patient_history_df[["Patient", "Weeks", "FVC", "Percent"]]

    if verbose:
        print(f'Patient {patient}; max: {chunked_slices.max()}, min: {chunked_slices.min()}')


    return chunked_slices, relevant_side_info

def read_in_data(data_dir="./data/train/", img_px_size=32, slice_count=8, verbose=False):
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
            img_data, patient_history = process_data(patient, patient_history_df, img_px_size=IMG_PX_SIZE, hm_slices=SLICE_COUNT, verbose=verbose)
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
    img_px_size = 64
    slice_count = 8
    patient_data = read_in_data(data_dir="./data/train/", img_px_size=img_px_size, slice_count=slice_count, verbose=True)
    save_to_disk(patient_data, img_px_size=img_px_size, slice_count=slice_count)

if __name__ == "__main__":
    main()