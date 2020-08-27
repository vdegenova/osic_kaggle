import os                                   # do directory ops
import math                                 # used for ceil ops
import datetime                             # used for saving with a datetime string
import pandas as pd                         # data analysis
import numpy as np                          # array ops
import matplotlib.pyplot as plt             # used for visualization in dev
import pydicom                              # for reading dicom files
from cv2 import resize, threshold, THRESH_OTSU     # image processing
from skimage import morphology, measure     # for lung masking
from sklearn.cluster import KMeans          # for lung masking
from tqdm import tqdm                       # for progress bars

def make_lungmask(img, display=False):
    '''
    - Standardize the pixel value by subtracting the mean and dividing by the standard deviation
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
    
    im_mean = np.mean(img)
    std = np.std(img)
    img = img-im_mean
    img = img/std
    # Find the average pixel value near the lungs to renormalize washed out images.
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    im_mean = np.mean(middle)  
    im_max = np.max(img)
    im_min = np.min(img)
    # To improve threshold finding, I'm moving the underflow and overflow on the pixel spectrum
    img[img==im_max]=im_mean
    img[img==im_min]=im_mean
    # Use Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.
    erosion = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(erosion,np.ones([15,15]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    labels = labels + 1 # add 1 to every element so that the background is no longer encoded as 0
    regions = measure.regionprops(labels) # for some reason ignores labels marked 0
    good_labels = []
    n_px = len(img.flatten())
    for prop in regions:
        B = prop.bbox # (min_row, min_col, max_row, max_col)
        # region width < 90% of img
        # region height < 90% of img
        # min row > 15%, max row < 85%
        # min col < 15%, max col < 85%
        # region area is > 0.1% of img
        if B[2]-B[0]<row_size*.90 and \
            B[3]-B[1]<col_size*.90 and \
            B[0]>row_size*.075 and B[2]<row_size*.925 and \
            B[1]>col_size*.075 and B[3]<col_size*925 and \
            prop.area/n_px*100 > 0.1:
            good_labels.append(prop.label) 
            #print(f'prop {prop.label} area: {np.round(prop.area/n_px*100,2)}, {B}')
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([5,5])) # one last dilation

    # final masked image
    masked_img = mask*img
    mask_var = np.round(np.var(mask), 4)
    n_labels = len(good_labels)

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
        ax[2, 1].imshow(masked_img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return masked_img, mask_var, n_labels


def make_lungmask_v2(img):

    # define new method using otsu's thresholding
    row_size= img.shape[0]
    col_size = img.shape[1]

    im_mean = np.mean(img)
    std = np.std(img)
    img = img-im_mean
    img = img/std
    # Find the average pixel value near the lungs to renormalize washed out images.
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    im_mean = np.mean(middle)  
    im_max = np.max(img)
    im_min = np.min(img)
    # To improve threshold finding, I'm moving the underflow and overflow on the pixel spectrum
    img[img==im_max]=im_mean
    img[img==im_min]=im_mean

    # Rescale pixels to 255
    img_2d = img.astype(float)                            
    im_min = np.min(img_2d) # many are not windowed and have vals < 0
    if im_min < 0:
        img_2d = img_2d - im_min
    img = (np.maximum(img_2d,0) / img_2d.max()) * 255.0

    # otsu's binarization thresholding
    thresh, thresh_img = threshold(img.astype('uint8'), np.min(img), np.max(img), THRESH_OTSU)
    # rescale to binary and invert
    thresh_img = np.where(thresh_img==0,1,0)

    ###########################################

    # now copy functionality from the old method to achieve a mask

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.
    erosion = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(erosion,np.ones([15,15]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    labels = labels + 1 # add 1 to every element so that the background is no longer encoded as 0
    regions = measure.regionprops(labels) # for some reason ignores labels marked 0
    good_labels = []
    n_px = len(img.flatten())
    for prop in regions:
        B = prop.bbox # (min_row, min_col, max_row, max_col)
        # region width < 90% of img
        # region height < 90% of img
        # min row > 15%, max row < 85%
        # min col < 15%, max col < 85%
        # region area is > 0.1% of img
        if B[2]-B[0]<row_size*.90 and \
            B[3]-B[1]<col_size*.90 and \
            B[0]>row_size*.075 and B[2]<row_size*.925 and \
            B[1]>col_size*.075 and B[3]<col_size*925 and \
            prop.area/n_px*100 > 0.1:
            good_labels.append(prop.label) 
            #print(f'prop {prop.label} area: {np.round(prop.area/n_px*100,2)}, {B}')
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([5,5])) # one last dilation

    # final masked image
    masked_img = mask*img
    mask_var = np.round(np.var(mask), 4)
    n_labels = len(good_labels)

    return masked_img, mask_var, n_labels


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
    slices = [dicom.pixel_array for dicom in dicoms]

    # lung masking
    masked_slices = []
    for each_slice in tqdm(slices):
        masked_img, mask_var, n_labels = make_lungmask_v2(each_slice)
        # check and see if this slice is worth keeping (contains lungs, did a good job of masking them)
        if mask_var > .04 and n_labels < 10:
            masked_slices.append(masked_img)

    # resize each pixel array - slices changed type to array here. Start as 512 x 512
    resized_masked_slices = []
    for each_slice in masked_slices:
        CROP_FACTOR = .2
        x, y = each_slice.shape
        xmin = int(np.floor(x*CROP_FACTOR))
        xmax = int(np.floor(x*(1-CROP_FACTOR)))
        ymin = int(np.floor(y*CROP_FACTOR))
        ymax = int(np.floor(y*(1-CROP_FACTOR)))
        resized_masked_slices.append(resize(np.array(each_slice[xmin:xmax, ymin:ymax]), (img_px_size, img_px_size)))
    # masked_slices = [resize(np.array(each_slice), (img_px_size, img_px_size)) for each_slice in masked_slices]

    # try opencv.resize on a different axis
    resized_masked_slices = np.array(resized_masked_slices)
    # initialze resized array, then modify each slice along second axis
    chunked_slices = np.zeros(shape=(hm_slices, img_px_size, img_px_size)) # (8, 64, 64)
    for i in range(chunked_slices.shape[1]):
        chunked_slices[:,i,:] = resize(resized_masked_slices[:,i,:], (img_px_size, hm_slices))
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

    patients = os.listdir("./data/train/") # list of training patient IDS (folders which contain DICOMS)
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
    now = datetime.datetime.now().isoformat(timespec='minutes')
    filestring = f'./data/processed_data/{data.shape[0]}-images-with_ids-{img_px_size}-{img_px_size}-{slice_count}-{now}.npy'
    print(f'saving to {filestring}')
    np.save(filestring, data)


def main():
    img_px_size = 64
    slice_count = 8
    patient_data = read_in_data(data_dir="./data/processed_data/", img_px_size=img_px_size, slice_count=slice_count, verbose=True)
    save_to_disk(patient_data, img_px_size=img_px_size, slice_count=slice_count)


if __name__ == "__main__":
    main()