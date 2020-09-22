import pandas as pd
import numpy as np

EXAMPLE_SUBMISSION_PATH = './data/sample_submission.csv'


def calculate_eval_function(row):
    sigma_clipped = max(row['Confidence'], 70)
    delta = min(np.abs(row['Truth']-row['FVC']), 1000)
    metric = (-np.sqrt(2) * delta)/sigma_clipped - \
        np.log(np.sqrt(2)*sigma_clipped)
    return metric


def evaluate_submission(eval_df):
    """
    This assumes you feed in a dataframe that looks exactly like the sample submission but that you also have a column
    called "Truth" that is the true target value to compare your "FVC" column against
    """
    log_likelihoods = []
    for _, row in eval_df.iterrows():
        laplace_log_likelihood = calculate_eval_function(row)
        log_likelihoods.append(laplace_log_likelihood)
    return np.mean(log_likelihoods)


def read_submission(filepath):
    if not filepath:
        print("No Input Filepath given")
        return None
    submission_df = pd.read_csv(filepath)
    return submission_df


def read_example_submission(filepath=EXAMPLE_SUBMISSION_PATH):
    """
    I'm using the sample submission here, and just adding noise to the prediction column to simulate the "true values"
    """
    submission_df = pd.read_csv(filepath)
    # creating noise with the same dimension as the dataset to simulate the "true values"
    mu, sigma = 0, 100
    noise = pd.Series(np.random.normal(
        mu, sigma, [len(submission_df), 1]).flatten())
    true_values = submission_df['FVC'] + noise
    submission_df['Truth'] = true_values
    return submission_df


def main():
    submission_filepath = './evaluation_output_training_data_q90.csv'
    submission_df = read_submission(submission_filepath)

    ### use this to try the sample submission ###
    # submission_df = read_example_submission()

    metric = evaluate_submission(submission_df)
    print(f'Laplace Log Likelihood: {metric}')


if __name__ == "__main__":
    main()
