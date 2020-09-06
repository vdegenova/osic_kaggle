import os  # do directory ops
import math  # used for ceil ops
import datetime  # used for saving with a datetime string
import pandas as pd  # data analysis
import numpy as np  # array ops
import matplotlib.pyplot as plt  # used for visualization in dev
import pydicom  # for reading dicom files
from cv2 import resize, threshold, THRESH_OTSU  # image processing
from skimage import morphology, measure  # for lung masking
from sklearn.cluster import KMeans  # for lung masking
from tqdm import tqdm  # for progress bars
from scipy.ndimage import gaussian_filter  # for lungmasking


def find_interior_ind(arr):
    # helper function for find_interior_ind. find border
    border_val = arr[0]
    borderless_start_ind = np.where(arr != border_val)[0][0]
    borderless_end_ind = np.where(arr != border_val)[0][-1]
    return borderless_start_ind + 1, borderless_end_ind - 1


def custom_trim(im):
    # returns an image with its border removed, if there is one
    mid_inds = np.array(im.shape) // 2
    middle_row = im[mid_inds[0], :]
    middle_col = im[:, mid_inds[1]]
    left, right = find_interior_ind(middle_row)
    top, bot = find_interior_ind(middle_col)

    # now, check if that entire column for left and right is monochrome
    if len(np.unique(im[:left])) > 1:
        left = 0 
    if len(np.unique(im[:right])) > 1:
        right = im.shape[0]
    if len(np.unique(im[top:])) > 1:
        top = 0
    if len(np.unique(im[bot:])) > 1:
        bot = im.shape[-1]
    return im[top:bot, left:right]


def transform_to_hu(img, rescale_slope, rescale_intercept):

    # convert ouside pixel-values to air:
    # I'm using <= -1000 to be sure that other defaults are captured as well
    hu_rescaled_img = img.copy()

    hu_rescaled_img[hu_rescaled_img <= -1000] = 0

    # convert to HU
    if rescale_slope != 1:
        hu_rescaled_img = rescale_slope * hu_rescaled_img.astype(np.float64)
        hu_rescaled_img = hu_rescaled_img.astype(np.int16)

    hu_rescaled_img = np.add(hu_rescaled_img, rescale_intercept, casting="safe")

    return hu_rescaled_img


def set_manual_window(hu_image, custom_center=-500, custom_width=1000):
    w_image = hu_image.copy()
    min_value = custom_center - (custom_width / 2)
    max_value = custom_center + (custom_width / 2)
    w_image[w_image < min_value] = min_value
    w_image[w_image > max_value] = max_value
    return w_image


def lung_mask(img, manual_threshold=320):
    # Returns the masked portion of the image showing just the lungs
    # With -320 we are separating between lungs (-700) /air (-1000) and tissue with values close to water (0).

    blurred_img = gaussian_filter(img, sigma=1)

    binary_image = np.array(blurred_img > -manual_threshold, dtype=np.int8) + 1
    manually_thresholded = binary_image.copy()

    labels = measure.label(binary_image)

    background_label_1 = labels[0, 0]
    background_label_2 = labels[0, -1]
    background_label_3 = labels[-1, 0]
    background_label_4 = labels[-1, -1]

    # Fill the air around the person
    binary_image[background_label_1 == labels] = 2
    binary_image[background_label_2 == labels] = 2
    binary_image[background_label_3 == labels] = 2
    binary_image[background_label_4 == labels] = 2

    # Morph - closing =  dilation followed by erosion
    kernel = morphology.disk(4)
    binary_image = morphology.closing(binary_image, kernel)

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    masked_img = binary_image.copy() * img

    return masked_img, manually_thresholded


def crop_and_resize(img, crop_factor, img_px_size):
    x, y = img.shape
    xmin = int(np.floor(x * crop_factor))
    xmax = int(np.floor(x * (1 - crop_factor)))
    ymin = int(np.floor(y * crop_factor))
    ymax = int(np.floor(y * (1 - crop_factor)))
    img = resize(np.array(img[xmin:xmax, ymin:ymax]), (img_px_size, img_px_size))
    return img


def resize_volume(slices, hm_slices, img_px_size):
    """
    takes in a list of slices, then resized it into a common depth
    """
    # try opencv.resize on a different axis
    slices = np.array(slices)
    # initialze resized array, then modify each slice along second axis
    resized_slices = np.zeros(
        shape=(hm_slices, img_px_size, img_px_size)
    )  # (8, 64, 64)
    for i in range(resized_slices.shape[1]):
        resized_slices[:, i, :] = resize(slices[:, i, :], (img_px_size, hm_slices))
        # interpolation methods are: (https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/)
        # INTER_NEAREST – a nearest-neighbor interpolation
        # INTER_LINEAR – a bilinear interpolation (used by default)
        # INTER_AREA – resampling using pixel area relation.
        #   It may be a preferred method for image decimation, as it gives moire’-free results.
        #   But when the image is zoomed, it is similar to the INTER_NEAREST method.
        # INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
        # INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood

    return resized_slices


def process_patient(
    patient,
    patient_history_df,
    img_px_size=32,
    hm_slices=8,
    verbose=False,
    data_dir="./data/train/",
    crop_factor=0,
):
    """
    Function to read in all of the DICOMS in a patient dir, condense arrays into aggregated chunks
    INPUTS:
        patient (str): Patient ID. data_dir/<patient>/ contains DICOMS
        patient_history_df (pd.DataFrame): dataframe of tabular data associated with this patient
        img_px_size (int): resize source DICOM image array to square matrix of this shape
        hm_slices (int): number of chunks to output
        crop_factor (float): between 0 and 1, percent of image to remove from borders of image
    RETURNS:
        img_data (np.Array): (hm_slices, img_px_size, img_px_size) array for one patient
        patient_history (pd.DataFrame): Patient tabular data, filtered to critical columns
    """

    path = data_dir + patient
    dicoms = [pydicom.read_file(path + "/" + s) for s in os.listdir(path)]
    dicoms.sort(
        key=lambda x: int(x.ImagePositionPatient[2])
    )  # sorts DICOMS by caudial (ass) to cranial (head)

    slices = []
    for dicom in tqdm(dicoms):
        # grab information from dicoms for later
        img = dicom.pixel_array
        rescale_intercept = dicom.RescaleIntercept
        rescale_slope = dicom.RescaleSlope

        # rescale HU
        hu_scaled_img = transform_to_hu(
            img=img, rescale_slope=rescale_slope, rescale_intercept=rescale_intercept
        )
        # window
        windowed_img = set_manual_window(hu_scaled_img)
        # mask slice
        masked_img, _ = lung_mask(windowed_img)
        # resize to common dimensions, optionally center crop
        resized_img = crop_and_resize(
            masked_img, crop_factor=crop_factor, img_px_size=img_px_size
        )
        # add finished image to list of slices
        slices.append(resized_img)

    # combine all slices into a volume and reshape into common volume
    resized_volume = resize_volume(slices, hm_slices, img_px_size)

    relevant_side_info = patient_history_df[["Patient", "Weeks", "FVC", "Percent"]]

    if verbose:
        print(f"Patient {patient}")

    return resized_volume, relevant_side_info


def read_in_data(
    csv_path,
    patient_dir,
    img_px_size=32,
    slice_count=8,
    verbose=False,
):
    """
    Function to ___
    INPUTS:
        csv_path (str): file location of train csv
        img_px_size (int): resize source DICOM image array to square matrix of this shape
        hm_slices (int): number of chunks to condense each patient DICOMs into
    RETURNS:
        all_the_data (np.array): array of size (n_patients, 2), where each entry is [DICOM_reshape_array, patient_id]
    """

    patients = os.listdir(
        patient_dir
    )  # list of training patient IDS (folders which contain DICOMS)
    train_df = pd.read_csv(csv_path)  # list of training patient tabular data

    IMG_PX_SIZE = img_px_size
    SLICE_COUNT = slice_count

    error_log = []
    all_the_data = []

    for num, patient in enumerate(patients):
        if num % 10 == 0:
            print("Patient:" + str(num))
        patient_history_df = train_df[train_df.Patient == patient].sort_values(
            by="Weeks"
        )

        try:
            img_data, patient_history = process_patient(
                patient,
                patient_history_df,
                img_px_size=IMG_PX_SIZE,
                hm_slices=SLICE_COUNT,
                verbose=verbose,
                data_dir=patient_dir,
            )
            patient_id = patient_history.Patient.iloc[0]
            all_the_data.append([img_data, patient_id])

        except Exception as e:
            print(patient, e)
            error_log.append((patient, e))
            continue

    return np.array(all_the_data, dtype=object)


def save_to_disk(data, img_px_size=32, slice_count=8, working_dir="./working/"):
    now = datetime.datetime.now().isoformat(timespec="minutes")
    filestring = f"{working_dir}{data.shape[0]}-images-with_ids-{img_px_size}-{img_px_size}-{slice_count}-{now}.npy"
    print(f"saving to {filestring}")
    np.save(filestring, data)
    return filestring


def main():
    LOCAL_RUN = True  # set this to False for submission!
    img_px_size = 64
    slice_count = 8
    local_csv_path = "./data/train.csv"
    kaggle_csv_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv"
    local_patient_dir = "./data/train/"
    kaggle_patient_dir = "./data/train/"

    patient_data = read_in_data(
        csv_path=local_csv_path if LOCAL_RUN else kaggle_csv_path,
        patient_dir=local_patient_dir if LOCAL_RUN else kaggle_patient_dir,
        img_px_size=img_px_size,
        slice_count=slice_count,
        verbose=True,
    )
    save_to_disk(patient_data, img_px_size=img_px_size, slice_count=slice_count)


if __name__ == "__main__":
    main()