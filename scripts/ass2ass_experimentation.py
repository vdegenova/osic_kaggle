import numpy  as np
import pandas as pd
import datetime
import pickle
from autoencoder import create_jesse_autoencoder, get_training__and_validation_data_and_patients, train_with_augmentation, encode_patients, merge_dicts
from regressor import load_pickled_encodings, get_pipeline_selectors, get_postproc_pipeline, create_dense_regressor, train_model
from inference import infer
from evaluation import evaluate_submission

def main():
    N_AUTOENCODER_EPOCHS = 50
    N_REGRESSOR_EPOCHS = 50

    # assume we've already pre processed the data for now (for experimentation)
    preprocessed_npy = './data/processed_data/158-images-with_ids-64-64-8-2020-08-26T21:54.npy'
    encoding_filepath = './data/processed_data/patient_ids_to_encodings_dict'

    # create and train autoencoder
    autoencoder, encoder = create_jesse_autoencoder()
    training_data, val_data, training_patients, val_patients = get_training__and_validation_data_and_patients(preprocessed_npy)
    train_with_augmentation(autoencoder, training_data, val_data, n_epochs=N_AUTOENCODER_EPOCHS, lr=1e-3)

    # Record final embedding for each patient {PatientID: flatten(embeddings)}
    encoded_training_patients = encode_patients(training_patients, training_data, encoder)
    encoded_val_patients = encode_patients(val_patients, val_data, encoder)
    all_encoded_patients = merge_dicts(encoded_training_patients, encoded_val_patients)

    # Save embeddings as pkl of dictionary
    now = datetime.datetime.now().isoformat(timespec='minutes')
    encoding_filepath += f"-{now}.pkl"
    print("{} Patients encoded.".format(len(all_encoded_patients)))
    print("Saving to {}.pkl".format(encoding_filepath))

    with open(encoding_filepath, 'wb') as f:
        pickle.dump(all_encoded_patients, f, pickle.HIGHEST_PROTOCOL)
    
    # train the regressor models

    # pass embedding + csv filepaths to function to unify into one dataframe
    csv_path = './data/train.csv'
    all_data = load_pickled_encodings(encoding_filepath, csv_path)
    X = all_data.drop(columns='FVC') # keep as a dataframe to pass to pipeline
    y = all_data[['FVC']]

    # get, run pipeline
    no_op_attrs, num_attrs, cat_attrs, encoded_attrs = get_pipeline_selectors()
    pipeline = get_postproc_pipeline(no_op_attrs, num_attrs, cat_attrs, encoded_attrs)
    X = pipeline.fit_transform(X).toarray() # returns scipy.sparse.csr.csr_matrix by default

    # Create regressora and  quantile regressor
    regressor = create_dense_regressor(n_input_dims=X.shape[1])
    quantile_regressor = create_dense_regressor(n_input_dims=X.shape[1], quantile_regression=True)

    # train both regressor models
    train_model(regressor, X, y, validation_split=.3, n_epochs=N_REGRESSOR_EPOCHS, suffix='og_regressor')
    train_model(quantile_regressor, X, y, validation_split=.3, n_epochs=N_REGRESSOR_EPOCHS, suffix='quantile_90')

    # run inference with the regressors on the training data and output to csv
    # generate FVC and confidence predictions
    preds = infer(regressor, X)
    quantile_preds = infer(quantile_regressor, X)

    y_arr = np.asarray(y['FVC'])
    patient_ids = np.asarray(all_data['Patient'])

    true_values = []
    patient_weeks = []
    FVCs = []
    confidences = []

    for patient, pred, truth, patient_id, q_pred in zip(X, preds, y_arr, patient_ids, quantile_preds):
        # print(f"ID: {patient_id} Weeks: {patient[0]} Pred: {pred[0]}, Truth: {truth}")
        patient_week = f"{patient_id}_{patient[0]}"
        patient_weeks.append(patient_week)
        FVCs.append(pred[0])
        confidences.append(abs(q_pred[0]-pred[0]))
        true_values.append(truth)
    
    results_filename =f'./data/evaluation_output/evaluation_output_training_data_q90_{now}.csv'

    results_df = pd.DataFrame({'Patient_Week': patient_weeks, 'FVC': FVCs, 'Confidence': confidences, 'Truth': true_values})    
    print(f'Writing Results to {results_filename}')
    results_df.to_csv(results_filename, index=False)

    # now we can evaluate the results we just created and saved off
    metric = evaluate_submission(results_df)
    print(f'Laplace Log Likelihood: {metric}')

if __name__ == "__main__":
    main()