import numpy as np
import pickle
import datetime
import pandas as pd
import os
from pre_proc import read_in_data, save_to_disk
from pipelines import (
    get_pipeline_selectors,
    get_postproc_pipeline,
    load_pickled_encodings,
)

from toolbox import select_predictions
from evaluation import evaluate_submission
import efficientnet_sandbox as efns


def main():
    # flags to assist in kaggle versus development
    is_kaggle_submission = False  # is this a final submission run for kaggle
    do_preproc = False            # should we read from disc and perform preprocessing
    # is this run on training data (offline eval)
    eval_on_training = False

    # setting up the respective numpy files for training/test to use if do_preproc is false
    if do_preproc:
        preprocessed_training_npy = None
        preprocessed_test_npy = None
    else:
        preprocessed_training_npy = "/Users/vdegenova/Projects/osic_kaggle/working/170-images-with_ids-64-64-8-2020-09-08T10:59.npy"
        preprocessed_test_npy = "/Users/vdegenova/Projects/osic_kaggle/working/5-images-with_ids-64-64-8-2020-09-08T10:59.npy"

    if not do_preproc:
        assert(preprocessed_training_npy is not None), "preprocessed_training_npy must be set when do_preproc is False"

    if is_kaggle_submission:
        # Kaggle specific file paths
        working_dir = "/kaggle/working/"
        working_dir_test = "/kaggle/working/test"
        temp_dir = "/kaggle/temp/"
        training_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
        training_csv_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv"
        test_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/"
        test_csv_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv"
        output_filepath = "/kaggle/working/submission.csv"
    else:
        # local filepaths
        working_dir = "./working/"
        working_dir_test = "./working/test"
        temp_dir = "./temp"
        training_dir = "./data/train/"
        training_csv_dir = "./data/train.csv"
        test_dir = "./data/test/"
        test_csv_dir = "./data/test.csv"

        substr = ""
        if eval_on_training:
            substr += "training_only_"
        output_filepath = f"./working/{substr}submission.csv"

    # overall program flow (submission)
    # 1. pre process data                               (train and test data)
    # 2. train wide and deep model                      (training data)
    # 3. Run inference & Calculate standard deviations  (test data)
    # 4. generate output file                           (test data)

    # overall program flow (eval_on_training = true)
    # 1. pre process data                               (train data)
    # 2. train wide and deep model                      (train data)
    # 3. Run inference & Calculate standard deviations  (train data)
    # 4. generate output file                           (train data)
    # 4. evaluate output file                           (train data)

    # constants
    img_px_size = 224
    slice_count = None  # setting to None will not resize slices in z
    # Save a volume for each patient - generates 1 .npy file
    SAVE_PATIENT_VOLUMES = False
    # Save a {patient:masks} json - generates 1 .json file
    SAVE_MASKING_DICT = False
    # Save a slice mask for each slice - generates 32,000 .npy files in <working_dir>/patient_masks_<im_px_size>/
    SAVE_SLICE_MASKS = True
    N_WIDE_AND_DEEP_EPOCHS = 0
    # whether or not to keep trtaining data in memory for the wide and deep model
    in_memory = True

    ################################################
    # 1. pre process data  (train and test data)
    ################################################
    # we only read in data from disc if do_preproc is set to true, otherwise we use the numpy files that are set above
    if do_preproc:
        patient_training_volumes, patient_training_masking_dict = read_in_data(
            csv_path=training_csv_dir,
            patient_dir=training_dir,
            img_px_size=img_px_size,
            slice_count=slice_count,
            verbose=True,
            SAVE_PATIENT_VOLUMES=SAVE_PATIENT_VOLUMES,
            SAVE_MASKING_DICT=SAVE_MASKING_DICT,
            SAVE_SLICE_MASKS=SAVE_SLICE_MASKS,
            working_dir=working_dir
        )

        if not eval_on_training:
            patient_test_volumes, patient_test_masking_dict = read_in_data(
                csv_path=test_csv_dir,
                patient_dir=test_dir,
                img_px_size=img_px_size,
                slice_count=slice_count,
                verbose=True,
                SAVE_PATIENT_VOLUMES=SAVE_PATIENT_VOLUMES,
                SAVE_MASKING_DICT=SAVE_MASKING_DICT,
                SAVE_SLICE_MASKS=SAVE_SLICE_MASKS,
                working_dir=working_dir_test
            )

    ################################################
    # 2. train wide and deep model  (training data)
    ################################################
    # Load masked images into datagenerators
    training_generator, validation_generator, tab_pipeline, fvc_pipeline = efns.load_training_datagenerators(
        LOCAL_PATIENT_MASKS_DIR=os.path.join(
            working_dir, f"patient_masks_{img_px_size}/"),
        LOCAL_PATIENT_TAB_PATH=training_csv_dir,
        in_memory=in_memory
    )

    if not eval_on_training:
        test_generator = efns.load_testing_datagenerator(
            LOCAL_PATIENT_MASKS_DIR=os.path.join(
                working_dir_test, f"patient_masks_{img_px_size}/"),
            LOCAL_PATIENT_TAB_PATH=test_csv_dir,
            tab_pipeline=tab_pipeline,
            in_memory=in_memory)

    model = efns.build_wide_and_deep()
    model.summary()

    # train model
    efns.train_model(model=model,
                     training_generator=training_generator,
                     validation_generator=validation_generator,
                     n_epochs=N_WIDE_AND_DEEP_EPOCHS
                     )

    ################################################
    # 3. Run inference & Calculate standard deviations  (test data)
    ################################################

    # select_predictions takes a trained model and a data generator
    # It uses eval_func to select a specific DICOM-prediction per patient-week
    # It uses conf_func to calculate the confidence of that prediction @ patient-week granularity
    # It expects the data generators to return batches @ 1 patient-week granularity and to generate all required batches
    # It expects the data generator to return iterables of [patient_id, week, [model-compliant-batch]]
    # It returns a pd.DataFrame in the submission format

    if eval_on_training:
        results_df = select_predictions(
            model, training_generator, eval_func="mean", conf_func="std", verbose="True")
    else:
        results_df = select_predictions(
            model, test_generator, eval_func="mean", conf_func="std", verbose="True")

    ################################################
    # 4. generate output file
    ################################################

    if eval_on_training:
        # add TRUTH column to results csv returned in 3.
        pass

    print(results_df.head())
    print(f"Writing Results to {output_filepath}")
    results_df.to_csv(output_filepath, index=False)

    ################################################
    # 8. evaluate output file if necessary
    ################################################
    if eval_on_training:
        metric = evaluate_submission(results_df)
        print(f'Laplace Log Likelihood: {metric}')


if __name__ == "__main__":
    main()
