import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow.keras.backend as K
import keras.losses


# custom modules
from post_proc import get_pipeline_selectors, get_postproc_pipeline, load_pickled_encodings

def tilted_loss(y, f):
    q = 0.9
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def load_model(filename='./regressor_model_D_0.5_LogCosh'):
    # keras.losses.custom_loss = tilted_loss
    model = keras.models.load_model(filename, custom_objects={'tilted_loss': tilted_loss})
    return model

def infer(model, x):
    preds = model.predict(x)
    return preds


def main():
    regressor = load_model()
    q90_model = load_model('./regressor_model_quantile_90_2020-08-03 20:57:04.830876')

    embeddings_path = './data/processed_data/patient_ids_to_encodings_dict-2020-07-26 13:50:14.433337.pkl'
    csv_path = './data/train.csv'
    all_data = load_pickled_encodings(embeddings_path, csv_path)
    X = all_data.drop(columns='FVC') # keep as a dataframe to pass to pipeline
    y = all_data[['FVC']]

    no_op_attrs, num_attrs, cat_attrs, encoded_attrs = get_pipeline_selectors()
    pipeline = get_postproc_pipeline(no_op_attrs, num_attrs, cat_attrs, encoded_attrs)
    X = pipeline.fit_transform(X).toarray() # returns scipy.sparse.csr.csr_matrix by default

    preds = infer(regressor, X)
    q90_preds = infer(q90_model, X)

    y_arr = np.asarray(y['FVC'])

    patient_ids = np.asarray(all_data['Patient'])

    true_values = []
    patient_weeks = []
    FVCs = []
    confidences = []
    CONF_VALUE = 100

    for patient, pred, truth, patient_id, q90_pred in zip(X, preds, y_arr, patient_ids, q90_preds):
        # print(f"ID: {patient_id} Weeks: {patient[0]} Pred: {pred[0]}, Truth: {truth}")
        patient_week = f"{patient_id}_{patient[0]}"
        patient_weeks.append(patient_week)
        FVCs.append(pred[0])
        confidences.append(abs(q90_pred[0]-pred[0]))
        true_values.append(truth)
    
    results_df = pd.DataFrame({'Patient_Week': patient_weeks, 'FVC': FVCs, 'Confidence': confidences, 'Truth': true_values})    
    results_df.to_csv('./training_data_output_q90.csv', index=False)
    

if __name__ == "__main__":
    main()