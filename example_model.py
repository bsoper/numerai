#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.metrics import log_loss as ll
from sklearn.grid_search import GridSearchCV as GS
from sklearn.metrics import make_scorer as MS


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = training_data['target']
    X = training_data.drop('target', axis=1)
    t_id = prediction_data['t_id']
    x_prediction = prediction_data.drop('t_id', axis=1)

    x_train, x_test, y_train, y_test = tts(X,Y, test_size=0.15, random_state=0)
    
    parameters = {
        "loss": ["log", "modified_huber"],
        "penalty": ["none", "l2", "l1", "elasticnet"],
        "alpha": [0.0001, 0.0003, 0.0009, 0.00003, 0.00001],
        "fit_intercept": [True, False],
        "n_iter": [5, 10],
        "verbose":[1]
    }

    best_params = {
       'alpha':[0.0002, 0.0003, 0.0004], 'average':[False, True], 'class_weight':[None, 'balanced'], 'epsilon':[0.1, 0.3, 0.9, 0.03, 0.01],
       'eta0':[0.0], 'fit_intercept':[False], 'l1_ratio':[0.15, 0.3, 0.45],
       'learning_rate':['optimal'], 'loss':['log'], 'n_iter':[5], 'n_jobs':[4],
       'penalty':['l1'], 'power_t':[0.5], 'random_state':[None], 'shuffle':[True],
       'verbose':[1], 'warm_start':[False]
    }

    # This is your model that will learn to predict
    scorer = MS(ll, greater_is_better=False, needs_proba=True, needs_threshold=False)

    grid_obj = GS(SGD(), best_params, scoring=scorer)

    print("Training....")
    # Your model is trained on the numerai_training_data
    grid_fit = grid_obj.fit(x_train, y_train)
    best_model = grid_fit.best_estimator_
    print(best_model)

    best_model.fit(x_train, y_train)
    y_predict = best_model.predict_log_proba(x_test)
    print(y_predict.shape)
    print(y_test.shape)
    print("Score", ll(y_test, y_predict))

    print("Predicting...")
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = best_model.predict_proba(x_prediction)
    results = y_prediction[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(t_id).join(results_df)

    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions.csv", index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
