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

from evaluation import evaluate_submission
import efficientnet_sandbox as efns


def main():
    # flags to assist in kaggle versus development
    is_kaggle_submission = False  # is this a final submission run for kaggle
    do_preproc = False            # should we read from disc and perform preprocessing
    eval_on_training = True       # is this run on training data (offline eval)

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
        temp_dir = "/kaggle/temp/"
        training_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
        training_csv_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv"
        test_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/"
        test_csv_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv"
        encoding_filepath = f"{working_dir}patient_ids_to_encodings_dict"
        output_filepath = "/kaggle/working/submission.csv"
    else:
        # local filepaths
        working_dir = "./working/"
        temp_dir = "./temp"
        training_dir = "./data/train/"
        training_csv_dir = "./data/train.csv"
        test_dir = "./data/test/"
        test_csv_dir = "./data/test.csv"
        encoding_filepath = f"{working_dir}patient_ids_to_encodings_dict"

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
    N_WIDE_AND_DEEP_EPOCHS = 1000

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
                working_dir=working_dir
            )

    ################################################
    # 2. train wide and deep model  (training data)
    ################################################
    # Load masked images into datagenerators
    training_generator, validation_generator = efns.load_training_dataset(
        LOCAL_PATIENT_MASKS_DIR=os.path.join(working_dir, 'patient_masks_224'),
        LOCAL_PATIENT_TAB_PATH=training_csv_dir
    )

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

    # INSERT STU'S BOSS ASS DATA ENGINEERING FUNCTION HERE
    # this function takes in the patients we need to generate output on
    # it also takes in the trained model
    # it also takes in min_weeks, and max_weeks, the range we need to predict on
    # it runs inference on each slice independently and then calculates a standard deviation
    # on each patient-week granularity

    ################################################
    # 4. generate output file
    ################################################


# OLD

    ################################################
    # 6. generate predictions and confidence values (test or train data)
    # whether we're using test or train data here depends on the flag eval_on_training set above
    ################################################
    # first preprocess the test data for the regressor models

    encodings_to_load = None
    csv_to_load = None

    if not eval_on_training:
        encodings_to_load = testing_encoding_path
        csv_to_load = test_csv_dir
    else:
        encodings_to_load = training_encoding_path
        csv_to_load = training_csv_dir

    all_test_data = load_pickled_encodings(encodings_to_load, csv_to_load)

    if not eval_on_training:
        all_test_data = all_test_data.drop_duplicates(
            subset=["Patient"], keep="first")

        min_weeks = -12
        max_weeks = 133
        weeks = list(range(min_weeks, max_weeks + 1, 1))

        total_df = pd.concat([all_test_data] * len(weeks))

        new_weeks = []
        n_test_patients = all_test_data.Patient.nunique()
        for week in weeks:
            this_week_duped = [week] * n_test_patients
            new_weeks.extend(this_week_duped)

        total_df["Weeks"] = new_weeks
        full_test_data = total_df
    else:
        full_test_data = all_test_data
        new_weeks = full_test_data.Weeks.values

    if eval_on_training:
        y_test = full_test_data.FVC.values

    X_test = full_test_data.drop(
        columns="FVC"
    )  # keep as a dataframe to pass to pipeline

    # run the pipeline
    X_test = pipeline.transform(
        X_test
    ).toarray()  # using the previously fit pipeline here on the test data

    preds = infer(regressor, X_test)
    quantile_preds = infer(quantile_regressor, X_test)
    patient_ids = np.asarray(full_test_data["Patient"])

    ################################################
    # 7. generate output file
    ################################################
    patient_weeks = []
    FVCs = []
    confidences = []

    if eval_on_training:
        true_values = []

        for patient, pred, patient_id, q_pred, week, true_y in zip(
            X_test, preds, patient_ids, quantile_preds, new_weeks, y_test
        ):
            # print(f"ID: {patient_id} Weeks: {patient[0]} Pred: {pred[0]}, Truth: {truth}")
            patient_week = f"{patient_id}_{week}"
            patient_weeks.append(patient_week)
            FVCs.append(int(pred[0]))
            confidences.append(int(abs(q_pred[0] - pred[0])))
            true_values.append(true_y)

        results_df = pd.DataFrame(
            {"Patient_Week": patient_weeks, "FVC": FVCs,
                "Confidence": confidences, "Truth": true_values}
        )
    else:
        for patient, pred, patient_id, q_pred, week in zip(
            X_test, preds, patient_ids, quantile_preds, new_weeks
        ):
            # print(f"ID: {patient_id} Weeks: {patient[0]} Pred: {pred[0]}, Truth: {truth}")
            patient_week = f"{patient_id}_{week}"
            patient_weeks.append(patient_week)
            FVCs.append(int(pred[0]))
            confidences.append(int(abs(q_pred[0] - pred[0])))

        results_df = pd.DataFrame(
            {"Patient_Week": patient_weeks, "FVC": FVCs, "Confidence": confidences}
        )

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
