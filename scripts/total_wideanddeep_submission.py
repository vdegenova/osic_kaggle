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
import shutil

# import sys
# import subprocess
# subprocess.check_call([sys.executable,
#                        '-m', 
#                        'pip', 
#                        'install', 
#                        '../input/kerasapplications/keras-team-keras-applications-3b180cb', 
#                        '-f', 
#                        './', 
#                        '--no-index'])
# subprocess.check_call([sys.executable,
#                        '-m', 
#                        'pip', 
#                        'install', 
#                        '../input/efficientnet/efficientnet-1.1.0', 
#                        '-f', 
#                        './', 
#                        '--no-index'])

from toolbox import select_predictions
# from evaluation import evaluate_submission
import efficientnet_sandbox as efns
import tensorflow as tf


def main():
    # flags to assist in kaggle versus development
    is_kaggle_submission = False  # is this a final submission run for kaggle
    do_preproc = True            # should we read from disc and perform preprocessing
    # is this run on training data (offline eval)
    eval_on_training = False

    # GPU check
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

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
        temp_dir = "../temp"
        training_dir = "../data/train/"
        training_csv_dir = "../data/train.csv"
        test_dir = "../data/test/"
        test_csv_dir = "../data/test.csv"

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
    N_WIDE_AND_DEEP_EPOCHS = 200
    # whether or not to keep trtaining data in memory for the wide and deep model
    in_memory = True

    ################################################
    # 1. pre process data  (train and test data)
    ################################################
    # we only read in data from disc if do_preproc is set to true, otherwise we use the numpy files that are set above
    if do_preproc:
        patient_training_volumes, patient_training_masking_dict, trapz_vol_dict_train = read_in_data(
            csv_path=training_csv_dir,
            patient_dir=training_dir,
            img_px_size=img_px_size,
            slice_count=slice_count,
            verbose=True,
            SAVE_PATIENT_VOLUMES=SAVE_PATIENT_VOLUMES,
            SAVE_MASKING_DICT=SAVE_MASKING_DICT,
            SAVE_SLICE_MASKS=SAVE_SLICE_MASKS,
            working_dir=working_dir,
            SAVE_TRAPEZOID_VOLUMES=True
        )

        # load train.csv and add calculated column, save as new
        df = pd.read_csv(training_csv_dir)
        df['TRAPZ_VOL'] = np.nan
        df['TRAPZ_VOL'] = df['Patient'].map(trapz_vol_dict_train)

        # this block instead saves train_mod.csv to the working_dir
        filename_ext_tuple_train = os.path.splitext(os.path.split(training_csv_dir)[-1])
        filestring_train = os.path.join(working_dir, filename_ext_tuple_train[0]+'_mod'+filename_ext_tuple_train[1])
        df.to_csv(filestring_train, index=False)

        training_csv_dir = filestring_train
    else:
        training_csv_dir = os.path.splitext(
            training_csv_dir)[0] + '_mod' + os.path.splitext(training_csv_dir)[1]

    if not eval_on_training:
        patient_test_volumes, patient_test_masking_dict, trapz_vol_dict_test = read_in_data(
            csv_path=test_csv_dir,
            patient_dir=test_dir,
            img_px_size=img_px_size,
            slice_count=slice_count,
            verbose=True,
            SAVE_PATIENT_VOLUMES=SAVE_PATIENT_VOLUMES,
            SAVE_MASKING_DICT=SAVE_MASKING_DICT,
            SAVE_SLICE_MASKS=SAVE_SLICE_MASKS,
            working_dir=working_dir_test,
            SAVE_TRAPEZOID_VOLUMES=True
        )

        # load train.csv and add calculated column, save as new
        df = pd.read_csv(test_csv_dir)
        df['TRAPZ_VOL'] = np.nan
        df['TRAPZ_VOL'] = df['Patient'].map(trapz_vol_dict_test)

        # this block instead saves train_mod.csv to the working_dir
        filename_ext_tuple_test = os.path.splitext(os.path.split(test_csv_dir)[-1])
        filestring_test = os.path.join(working_dir, filename_ext_tuple_test[0]+'_mod'+filename_ext_tuple_test[1])
        df.to_csv(filestring_test, index=False)

        test_csv_dir = filestring_test
    else:
        test_csv_dir = os.path.splitext(
            test_csv_dir)[0] + '_mod' + os.path.splitext(test_csv_dir)[1]

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

    if is_kaggle_submission:
        model = efns.build_wide_and_deep(
            weights='../input/efficientnet-weights-for-keras/noisy-student/notop/efficientnet-b0_noisy-student_notop.h5'
        )
    else:
        model = efns.build_wide_and_deep()
    model.summary()

    # train model
    efns.train_model(model=model,
                     training_generator=training_generator,
                     validation_generator=validation_generator,
                     n_epochs=N_WIDE_AND_DEEP_EPOCHS,
                     with_callbacks=False
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

    results_df.FVC = pd.Series(np.squeeze(fvc_pipeline.transformer_list[-1][-1].steps[-1][-1].inverse_transform(
        np.array([results_df.FVC.values]))))

    results_df.Confidence = pd.Series(np.squeeze(fvc_pipeline.transformer_list[-1][-1].steps[-1][-1].inverse_transform(
        np.array([results_df.Confidence.values]))))

    ################################################
    # 4. generate output file
    ################################################

    if eval_on_training:
        # add TRUTH column to results csv returned in 3.
        pass

    print(results_df.head())
    print(f"Writing Results to {output_filepath}")
    results_df.to_csv(output_filepath, index=False)
    # delete patient mask npy files...
    # delete npy masks
    shutil.rmtree(
        os.path.join(
            working_dir, f"patient_masks_{img_px_size}/")
    )
    shutil.rmtree(
        os.path.join(
            working_dir, 'test', f"patient_masks_{img_px_size}/")
    )

    ################################################
    # 8. evaluate output file if necessary
    ################################################
    # if eval_on_training:
    # metric = evaluate_submission(results_df)
    # print(f'Laplace Log Likelihood: {metric}')


if __name__ == "__main__":
    main()
