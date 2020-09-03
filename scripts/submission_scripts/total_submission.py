import numpy as np
import pickle
import datetime
import pandas as pd
from pre_proc2 import read_in_data, save_to_disk
from autoencoder import (
    create_jesse_autoencoder,
    get_training__and_validation_data_and_patients,
    train_with_augmentation,
    encode_patients,
    merge_dicts,
    get_test_data_and_patients,
)
from pipelines import (
    get_pipeline_selectors,
    get_postproc_pipeline,
    load_pickled_encodings,
)
from regressor import create_dense_regressor, train_model
from inference import infer


def main():

    is_kaggle_submission = False

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
        output_filepath = "./working/submission.csv"

        # preprocessed_training_npy = "/Users/vdegenova/Projects/osic_kaggle/working/158-images-with_ids-64-64-8-2020-09-01T21:35.npy"
        # preprocessed_test_npy = "/Users/vdegenova/Projects/osic_kaggle/working/5-images-with_ids-64-64-8-2020-09-01T21:35.npy"

    # overall program flow
    # 1. read in data                               (train and test)
    # 2. pre process data                           (train and test)
    # 3. train autoencoder                          (training data)
    # 4. embed images                               (train and test)
    # 5. train regressor models                     (training data)
    # 6. generate predictions and confidence values (test data)
    # 7. generate output file                        (test data)

    # constants
    img_px_size = 64
    slice_count = 8
    N_AUTOENCODER_EPOCHS = 1
    N_REGRESSOR_EPOCHS = 1

    ################################################
    # 1. read in data (train and test)
    ################################################
    patient_training_data = read_in_data(
        data_dir=training_dir,
        img_px_size=img_px_size,
        slice_count=slice_count,
        verbose=True,
        csv_file=training_csv_dir,
    )
    patient_test_data = read_in_data(
        data_dir=test_dir,
        img_px_size=img_px_size,
        slice_count=slice_count,
        verbose=True,
        csv_file=test_csv_dir,
    )

    ################################################
    # 2. pre process dicom data (train and test)
    ################################################
    preprocessed_training_npy = save_to_disk(
        patient_training_data,
        img_px_size=img_px_size,
        slice_count=slice_count,
        working_dir=working_dir,
    )
    preprocessed_test_npy = save_to_disk(
        patient_test_data,
        img_px_size=img_px_size,
        slice_count=slice_count,
        working_dir=working_dir,
    )
    del patient_training_data
    del patient_test_data

    ################################################
    # 3. train autoencoder (training data)
    ################################################
    autoencoder, encoder = create_jesse_autoencoder(lr=1e-4)
    (
        training_data,
        val_data,
        training_patients,
        val_patients,
    ) = get_training__and_validation_data_and_patients(preprocessed_training_npy)
    train_with_augmentation(
        autoencoder, training_data, val_data, n_epochs=N_AUTOENCODER_EPOCHS
    )

    ################################################
    # 4. embed dicom images (train and test)
    ################################################
    # format is like {PatientID: flatten(embeddings)}
    encoded_training_patients = encode_patients(
        training_patients, training_data, encoder
    )
    encoded_val_patients = encode_patients(val_patients, val_data, encoder)
    all_encoded_training_patients = merge_dicts(
        encoded_training_patients, encoded_val_patients
    )

    test_data, test_patients = get_test_data_and_patients(preprocessed_test_npy)
    encoded_test_patients = encode_patients(test_patients, test_data, encoder)

    # Save embeddings as pkl of dictionary
    now = datetime.datetime.now().isoformat(timespec="minutes")
    training_encoding_path = encoding_filepath + f"-training-{now}.pkl"
    print("{} Training Patients encoded.".format(len(all_encoded_training_patients)))
    print("Saving to {}.pkl".format(training_encoding_path))

    with open(training_encoding_path, "wb") as f:
        pickle.dump(all_encoded_training_patients, f, pickle.HIGHEST_PROTOCOL)
    del all_encoded_training_patients

    testing_encoding_path = encoding_filepath + f"-test-{now}.pkl"
    print("{} Test Patients encoded.".format(len(encoded_test_patients)))
    print("Saving to {}.pkl".format(testing_encoding_path))

    with open(testing_encoding_path, "wb") as f:
        pickle.dump(encoded_test_patients, f, pickle.HIGHEST_PROTOCOL)
    del encoded_test_patients

    ################################################
    # 5. train regressor models (training data)
    ################################################
    # pass embedding + csv filepaths to function to unify into one dataframe
    all_data = load_pickled_encodings(training_encoding_path, training_csv_dir)
    X = all_data.drop(columns="FVC")  # keep as a dataframe to pass to pipeline
    y = all_data[["FVC"]]

    # get, run pipeline
    no_op_attrs, num_attrs, cat_attrs, encoded_attrs = get_pipeline_selectors()
    pipeline = get_postproc_pipeline(no_op_attrs, num_attrs, cat_attrs, encoded_attrs)
    X = pipeline.fit_transform(
        X
    ).toarray()  # returns scipy.sparse.csr.csr_matrix by default

    # Create regressora and  quantile regressor
    regressor = create_dense_regressor(n_input_dims=X.shape[1])
    quantile_regressor = create_dense_regressor(
        n_input_dims=X.shape[1], quantile_regression=True
    )

    # train both regressor models
    train_model(
        regressor,
        X,
        y,
        validation_split=0.3,
        n_epochs=N_REGRESSOR_EPOCHS,
        suffix="og_regressor",
    )
    train_model(
        quantile_regressor,
        X,
        y,
        validation_split=0.3,
        n_epochs=N_REGRESSOR_EPOCHS,
        suffix="quantile_90",
    )

    ################################################
    # 6. generate predictions and confidence values (test data)
    ################################################
    # first preprocess the test data for the regressor models

    all_test_data = load_pickled_encodings(testing_encoding_path, test_csv_dir)

    min_weeks = -12
    max_weeks = 133
    weeks = list(range(min_weeks, max_weeks + 1, 1))

    all_test_data = all_test_data.drop_duplicates(subset=["Patient"], keep="first")

    total_df = pd.concat([all_test_data] * len(weeks))

    new_weeks = []
    n_test_patients = all_test_data.Patient.nunique()
    for week in weeks:
        this_week_duped = [week] * n_test_patients
        new_weeks.extend(this_week_duped)

    total_df["Weeks"] = new_weeks
    full_test_data = total_df

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


if __name__ == "__main__":
    main()