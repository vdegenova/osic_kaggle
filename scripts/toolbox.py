"""Many useful functions"""

import os
import pydicom
import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict


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
            slices = [pydicom.read_file(f"{patient_path}/{s}")
                      for s in os.listdir(patient_path)]
            if sort_slices:
                # sorts DICOMS by caudial (ass) to cranial (head)
                slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            patient_dicoms[pid] = slices
        except Exception as err:  # pylint: disable=broad-except
            print("Something bad happened!")
            print(err)
            print(f"Skipping patient {pid} & continuing...")
            continue  # Keep processing other DICOMs

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


def select_predictions(model, data_generator, eval_func="mean", eval_lambda=None, conf_func="std", conf_lambda=None, verbose=False):
    """Iteratively uses :model: and :data_generator: to make FVC predictions on sets of patient DICOMs.
        Uses statistical measures to select a scalar prediction & calculate the inference confidence

    :param model: An inference model with a model.predict interface
    :type model: any interface-compliant class, required
    :param data_gen: A generator that iteratively returns batches of data in a (patient_id, week, batch) format.
    :type data_gen: generator, required
    :param eval_func: Which statistical measure to use to select DICOM-predictions for a patient
        This parameter is ignored if eval_lambda is passed.
    :type eval_func: str, optional
    :param eval_lambda: A lambda function to use to select a DICOM-predection from an array of predictions.
        Must support batch operations. If passed, :eval_func: is ignored.
    :type eval_lambda: func, optional
    :param conf_func: Which statistical function to use to calculate prediciton confidence
    :type conf_func: str, optional
    :param conf_lambda: A lambda function to use to generate the confidence of a DICOM-predection from an array of predictions.
        Must support batch operations. If passed, :conf_func: is ignored.
    :type conf_lambda: func, optional
    :param verbose: Controls function verbosity while searching
    :type verbose: bool, optional
    :return: Returns a data frame containing a prediction row for each unique patient-week
    :rtype: pd.DataFrame
    """

    eval_functions = {
        "mean": np.mean,
        "median": np.median,
    }

    if eval_lambda is not None:
        eval_func = eval_lambda
    else:
        eval_func = eval_functions[eval_func]

    conf_functions = {
        "std": np.std,
    }

    if conf_lambda is not None:
        conf_func = conf_lambda
    else:
        conf_func = conf_functions[conf_func]

    if verbose:
        print(
            f"Using eval function: {eval_func} and confidence function: {conf_func}")

    patient_predictions = pd.DataFrame(
        [], columns=["Patient_Week", "FVC", "Confidence"])

    # Each batch is an array of [biographics, DICOM encodings] for a single week of a single patient
    for patient_id, week, batch in data_generator:
        # Each batch-prediction returns an array of predictions, 1 per DICOM encoding
        predictions = model.predict_on_batch(batch)
        # Select which DICOM-prediction to use for this patient-week
        selection = eval_func(predictions)
        # Generate the confidence of the DICOM-prediction for the patient-week
        conf = conf_func(predictions)

        if verbose:
            print(
                f"Selected FVC prediction for patient {patient_id} on week {week}: {selection}, confidence {conf}")

        patient_week = patient_id + week
        patient_predictions.append([patient_week, selection, conf])

    return patient_predictions
