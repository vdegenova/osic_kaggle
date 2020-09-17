"""Many useful functions"""

import os
import pydicom
import tqdm
import pandas as pd
import numpy as np

def get_patient_data(csv_path, patient_ids=None):
    """Returns the medical data associated with the provided patient_id,
        or for all patients if not provided.

    :param csv_path: Absolute path to the csv file continaing patient data, e.g. 'train.csv'
    :type csv_path: str, required
    :param patient_id: A list of patient IDs as strings used to select patient data.
        If None, selects all patients.
    :type patient_id: list, optional
    :return: DataFrame containing medical data of the selected patients
    :rtype: pandas.DataFrame
    """
    data = pd.read_csv(csv_path)
    if patient_ids is None:
        return data
    return data[data['Patient'].isin(patient_ids)]


def get_patient_dicoms(dicom_path, patient_ids=None, sort_slices=True):
    """Returns the DICOM images associated with the provided patient_id,
        or for all patients if not provided.

    :param dicom_path: Absolute path to the directory containing the subdirs of patient DICOM images
    :type dicom_path: str, required
    :param patient_id: A list of patient IDs as strings used to select patient DICOM images.
        If None, selects all patients.
    :type patient_id: list, optional
    :param sort_slices: Bool indicating to sort the DICOM slices by z-index, caudial to cranial
    :type sort_slices: bool, optional
    :return: Dictionary containing the DICOM images of each selected patient,
        where the keys are patient_id and the value is a list of DICOM slices.
    :rtype: dict
    """
    patient_dicoms = {}
    if patient_ids is None:
        # DICOMS are stored in dirs with patient_ids as names
        patient_ids = os.listdir(dicom_path)

    for pid in patient_ids:
        # Known issues with some DICOMs prevent processing; skip them
        try:
            patient_path = f"{dicom_path}/{pid}"
            slices = [pydicom.read_file(f"{patient_path}/{s}") for s in os.listdir(patient_path)]
            if sort_slices:
                # sorts DICOMS by caudial (ass) to cranial (head)
                slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            patient_dicoms[pid] = slices
        except Exception as err:  #pylint: disable=broad-except
            print("Something bad happened!")
            print(err)
            print(f"Skipping patient {pid} & continuing...")
            continue # Keep processing other DICOMs

    return patient_dicoms


def score_prediction(real, pred, conf, min_conf=70, max_err=1000):
    """Returns the bounded Laplace log likelihood based on prediction error and confidence

    :param real: The real value of the data
    :type real: float, required
    :param pred: The predicted value of the data
    :type pred: float, required
    :param conf: The stated confidence of the prediction
    :type conf: float, required
    :param min_conf: The lower bound of prediction confidence; represents min possible error
    :type min_conf: float, optional
    :param max_err: The upper bound of prediction error to limit scoring penalty
    :type max_err: float, optional
    :return: The LaPlace Log Likelihood based on prediction error and the bounded confidence
    :rtype: float
    """
    # Clipping defaults provided by competiton
    bounded_conf = np.maximum(conf, min_conf)
    bounded_err = np.minimum(np.abs(real - pred), max_err)
    return -(np.sqrt(2) * bounded_err / bounded_conf) - np.log(np.sqrt(2) * bounded_conf)

def search_best_confidence(model, csv_path, dicom_path, conf_band_size=10, verbose=False):
    """Iteratively uses :model: to make FVC predictions at different confidence levels
        to find the confidence that provides the best average laplace log likelihood for :model:

    :param model: An inference model with a model.predict interface
    :type model: any interface-compliant class, required
    :param csv_path: Absolute path to the csv file continaing patient data, e.g. 'train.csv'
    :type csv_path: str, required
    :param dicom_path: Absolute path to the directory containing the subdirs of patient DICOM images
    :type dicom_path: str, required
    :param conf_band_size: The size of confidence bands to test between 0 and 100.
        Defaults to 10 (confidence deciles), to predict at 10th, 20th, etc. percentiles
    :type conf_band_size: int, required
    :param verbose: Controls function verbosity while searching
    :type verbose: bool, optional
    :return: The confidence band with the highest average laplace log likelihood for :model:
    :rtype: int
    """

    # Load data
    average_scores = []
    patient_data = get_patient_data(csv_path)
    patient_dicoms = get_patient_dicoms(dicom_path)

    # Make predictions at each confidence band
    for conf in tqdm.tqdm(range(conf_band_size, 100, conf_band_size)):
        conf_scores = []
        # Use DICOM dict for keys to avoid patients with bad DICOMs
        for pid, dicoms in patient_dicoms.items():
            bio_data = patient_data[patient_data['Patient'] == pid]
            dicoms = np.expand_dims(dicoms, axis=-1)

            # Multiple training weeks for each patient - predict & score at each
            patient_scores = []
            for weekly_data in bio_data.itertuples():
                real_fvc = weekly_data.FVC
                # Vector of per-slice predictions
                fvc_predictions = model.predict([bio_data, dicoms]) 
                # Select the prediction that is the :conf:'th percentile of predictions
                pred_fvc = np.quantile(fvc_predictions, conf / 100) # q param must be between 0-1
                patient_scores.append(score_prediction(real_fvc, pred_fvc, conf))

            # Calc and add the patient's average prediction score for all slices
            patient_avg_score = np.mean(patient_scores)
            conf_scores.append(patient_avg_score)
            if verbose:
                print(
                    f"Mean prediction score for patient {pid} at conf % {conf}: {patient_avg_score}"
                )

        # Calc and add the overall average prediction score for the current conf percentile
        avg_conf_score = np.mean(conf_scores)
        average_scores.append(avg_conf_score)
        if verbose:
            print(
                f"Mean prediction score for all patients at conf % {conf}: {avg_conf_score}"
            )

    # Find the confidence with the best avg score for all patients - higher is better for LPL
    best_score_idx = (np.argmax(average_scores) + 1)
    best_confidence = conf_band_size * best_score_idx
    if verbose:
        print(f"Confidence with the highest average prediction score: {best_confidence}")

    return best_confidence
