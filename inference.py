import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# custom modules
from post_proc import get_pipeline_selectors, get_postproc_pipeline, load_pickled_encodings

def load_model(filename='./regressor_model_D_0.5_LogCosh'):
    model = keras.models.load_model(filename)
    return model

def infer(model, x):
    preds = model.predict(x)
    return preds

def main():
    regressor = load_model()

    embeddings_path = './data/processed_data/patient_ids_to_encodings_dict-2020-07-26 13:50:14.433337.pkl'
    csv_path = './data/train.csv'
    all_data = load_pickled_encodings(embeddings_path, csv_path)
    X = all_data.drop(columns='FVC') # keep as a dataframe to pass to pipeline
    y = all_data[['FVC']]

    no_op_attrs, num_attrs, cat_attrs, encoded_attrs = get_pipeline_selectors()
    pipeline = get_postproc_pipeline(no_op_attrs, num_attrs, cat_attrs, encoded_attrs)
    X = pipeline.fit_transform(X).toarray() # returns scipy.sparse.csr.csr_matrix by default

    preds = infer(regressor, X)

    y_arr = np.asarray(y['FVC'])

    patient_ids = np.asarray(all_data['Patient'])

    true_values = []
    predictions = []

    for patient, pred, truth, patient_id in zip(X, preds, y_arr, patient_ids):
        print(f"ID: {patient_id} Weeks: {patient[0]}, Patient: {patient[1]}, Percent: {patient[2]}")
        print(f'Pred: {pred[0]}, Truth: {truth}')

        predictions.append(pred[0])
        true_values.append(truth)
        if patient_id != patient_ids[0]:
            break
    
    plt.scatter(predictions, true_values)
    plt.show()
    
    

if __name__ == "__main__":
    main()