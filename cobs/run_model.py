#!/usr/bin/env python

"""
Process dataset, select model and run with parameters.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.utils import class_weight as cw
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, recall_score
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from src.main import read_model_config, evaluate, start_log
from src.data_loader import get_data
from src.models.keras_models import Residual, Perceptron, create_callbacks, MultilayerPerceptron
from keras.wrappers.scikit_learn import KerasClassifier
import xgboost as xgb


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')


def get_options():
    default_path = os.path.dirname(os.path.realpath(__file__)).replace("/cobs", "")
    parser = argparse.ArgumentParser(prog="Classification of biochemical sequences")
    parser.add_argument('select_model', nargs='+', help='name of the model, select from list in README'),
    parser.add_argument('dataset_path', nargs='+', default=default_path + '/tmp/dataset.csv', help='path to dataset'),
    parser.add_argument('--output', default=default_path + "/cobs/tmp/" + str(datetime.now()) + '/', help='path to output directory'),
    parser.add_argument('--configs', default=default_path + "/cobs/cobs/configs.ini", help='path to config file'),
    parser.add_argument('--n_iter', default=6, type=int, help='number of iterations in RandomizedSearchCV'),
    parser.add_argument('--n_jobs', default=-1, type=int, help='number of jobs'),
    parser.add_argument('--n_folds', default=5, type=int, help='number of splits in RandomizedSearchCV'),
    parser.add_argument('--patience', '-p', default=100, type=int, help='patience of fit'),
    parser.add_argument('--gridsearch', '-g', action='store_true', default=False, help='use gridsearch'),
    parser.add_argument('--experiments_file', '-e', default='etc/experiments.csv', help='where to write results of experiments')
    parser.add_argument('--length', '-l', default='256', type=int, help='maximum length of sequences'),
    parser.add_argument('--targets', '-t', default=0, type=int, help='set number of target column')
    return parser


def fingerprint(seq, length):
    """
    Create fingerprint of sequence.

    Params
    ------
    seq: string
        sequence from dataset
    length: list or False
        length of fingerprint

    Output
    ------
    f: list
        fingerprint of sequence
    """
    f = []
    for char in seq:
        if len(f) < int(length):
            f.append(ord(char))
    while len(f) < int(length):
        f.append(0)
    return f


def run(args_list, random_state=False, p_rparams=False):
    """
    Run experiment.
    
    Params
    ------
    args_list: options
    random_state: int
        random state of all
    p_rparams: list or False
        rparams from previous experiment

    Output
    ------
    Result of experiment in metrics:
        accuracy_test: string
        accuracy_train: string
        rec: list
        auc: string
        auc_val: string
        f1: string
    """
    scoring = "accuracy"
    time_start = datetime.now()
    if len(sys.argv) > 1:
        options = get_options().parse_args()
    else:
        options = get_options().parse_args(args_list)

    callbacks_list = create_callbacks(options.output, options.patience, "cobs")

    handler = logging.FileHandler(options.output + 'log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    epochs, rparams, gparams = read_model_config(options.configs, options.select_model[0])
    data = pd.read_csv(options.dataset_path[0])
    data = np.array(data)
    data = data.T

    labels = []
    features = []
    length = len(max(data[1], key=len))
    if length > options.length:
        length = options.length
    for d in data[1]:
        features.append(fingerprint(d, length))
    for l in data[2]:
        labels.append(int(l))

    # Dataset processing

    x_train, x, y_train, y = train_test_split(features, labels, test_size=0.2, random_state=random_state)
    x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.8, random_state=random_state)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    input_shape = int(length)
    output_shape = 1

    # Model selection

    if options.gridsearch and not p_rparams:
        # check if the number of iterations more then possible combinations
        keys = list(gparams.keys())
        n_iter = 1
        for k in keys:
            n_iter *= len(gparams[k])
        if options.n_iter > n_iter:
            options.n_iter = n_iter

        logger.info("GRID SEARCH")

        if options.select_model[0] == "logreg":
            model = RandomizedSearchCV(LogisticRegression(**rparams),
                                       gparams,
                                       n_iter=options.n_iter,
                                       n_jobs=options.n_jobs,
                                       cv=options.n_folds,
                                       verbose=10,
                                       scoring=scoring)
        elif options.select_model[0] == "knn":
            model = RandomizedSearchCV(KNeighborsClassifier(**rparams),
                                       gparams,
                                       n_iter=options.n_iter,
                                       n_jobs=options.n_jobs,
                                       cv=options.n_folds,
                                       verbose=10,
                                       scoring=scoring)
        elif options.select_model[0] == "xgb":
            model = RandomizedSearchCV(xgb.XGBClassifier(**rparams),
                                       gparams,
                                       n_iter=options.n_iter,
                                       n_jobs=options.n_jobs,
                                       cv=options.n_folds,
                                       verbose=10,
                                       scoring=scoring)
        elif options.select_model[0] == "svc":
            model = RandomizedSearchCV(SVC(**rparams),
                                       gparams,
                                       n_iter=options.n_iter,
                                       n_jobs=options.n_jobs,
                                       cv=options.n_folds,
                                       verbose=10,
                                       scoring=scoring)
        elif options.select_model[0] == "rf":
            model = RandomizedSearchCV(RandomForestClassifier(**rparams),
                                       gparams,
                                       n_iter=options.n_iter,
                                       n_jobs=options.n_jobs,
                                       cv=options.n_folds,
                                       verbose=10,
                                       scoring=scoring)
        elif options.select_model[0] == "if":
            model = RandomizedSearchCV(IsolationForest(**rparams),
                                       gparams,
                                       n_iter=options.n_iter,
                                       n_jobs=options.n_jobs,
                                       cv=options.n_folds,
                                       verbose=10,
                                       scoring=scoring)
        elif options.select_model[0] == "perceptron":
            search_model = KerasClassifier(build_fn=Perceptron,
                                           input_shape=input_shape,
                                           output_shape=output_shape)
            model = RandomizedSearchCV(estimator=search_model,
                                      param_distributions=gparams,
                                      n_jobs=options.n_jobs,
                                      cv=options.n_folds,
                                      n_iter=options.n_iter,
                                      verbose=10)
        elif options.select_model[0] == "mperceptron":
            search_model = KerasClassifier(build_fn=MultilayerPerceptron,
                                           input_shape=input_shape,
                                           output_shape=output_shape)
            model = RandomizedSearchCV(estimator=search_model,
                                      param_distributions=gparams,
                                      n_jobs=options.n_jobs,
                                      cv=options.n_folds,
                                      n_iter=options.n_iter,
                                      verbose=10)
        elif options.select_model[0] == "residual":
            search_model = KerasClassifier(build_fn=Residual,
                                           input_shape=input_shape,
                                           output_shape=output_shape)
            model = RandomizedSearchCV(estimator=search_model,
                                      param_distributions=gparams,
                                      n_jobs=options.n_jobs,
                                      cv=options.n_folds,
                                      n_iter=options.n_iter,
                                      verbose=10)

            #model = grid.fit(x_train, y_train)
            

        else:
            logger.info("Model is not found.")
            return 0, 0, 0, 0, 0, 0

        logger.info("FIT")
        history = model.fit(x_train, np.ravel(y_train))

    else:
        if p_rparams:
            rparams = p_rparams
        if options.select_model[0] == "logreg":
            rparams['verbose'] = 10
            model = LogisticRegression(**rparams)
        elif options.select_model[0] == "knn":
            model = KNeighborsClassifier(**rparams)
        elif options.select_model[0] == "svc":
            rparams['class_weight'] = "balanced"
            model = SVC(**rparams)
        elif options.select_model[0] == "rf":
            rparams['class_weight'] = "balanced"
            model = RandomForestClassifier(**rparams)
        elif options.select_model[0] == "if":
            model = IsolationForest(**rparams)
        elif options.select_model[0] == "xgb":
            model = xgb.XGBClassifier(**rparams)
        elif options.select_model[0] == "perceptron":
            model = Perceptron(input_shape,
                               output_shape,
                               activation=rparams.get("activation"),
                               loss=rparams.get("loss"),
                               metrics=rparams.get("metrics"),
                               optimizer=rparams.get("optimizer"),
                               learning_rate=rparams.get("learning_rate"),
                               momentum=rparams.get("momentum"),
                               init_mode=rparams.get("init_mode"))
        elif options.select_model[0] == "residual":
            model = Residual(input_shape,
                             output_shape,
                             activation_0=rparams.get("activation_0", 'softmax'),
                             activation_1=rparams.get("activation_0", 'softmax'),
                             activation_2=rparams.get("activation_0", 'softmax'),
                             loss=rparams.get("loss", 'binary_crossentropy'),
                             metrics=rparams.get("metrics", ['accuracy']),
                             optimizer=rparams.get("optimizer", 'Adam'),
                             learning_rate=rparams.get("learning_rate", 0.001),
                             momentum=rparams.get("momentum", 0.1),
                             init_mode=rparams.get("init_mode", 'uniform'),
                             dropout=rparams.get("dropout", 0),
                             layers=rparams.get("layers", 0))
            
        else:
            logger.info("Model is not found.")
            return 0, 0, 0, 0, 0, 0

        logger.info("FIT")

        if options.select_model[0] == "perceptron" or options.select_model[0] == "residual":
            history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size"),
                                epochs=epochs, shuffle=True, verbose=1, callbacks=callbacks_list)
        else:
            history = model.fit(x_train, y_train)

    if options.gridsearch and not p_rparams:
        rparams = model.best_params_
        score = pd.DataFrame(model.cv_results_)
    else:
        score = False
    
    accuracy_test, accuracy_train, rec, auc, auc_val, f1 = evaluate(logger, options, random_state, options.output, model, x_train,
                                                                    x_test, x_val, y_val, y_train, y_test, time_start, rparams, history,
                                                                    False, False, options.n_jobs, score)
    return accuracy_test, accuracy_train, rec, auc, f1, rparams


if __name__ == "__main__":
    args_list = ['logreg', 'data/dataset.csv']
    run(args_list)
