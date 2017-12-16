#!/usr/bin/env python

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
from sklearn.metrics import matthews_corrcoef, make_scorer
from src.main import create_callbacks, read_config, evaluate, start_log
from src.data_loader import get_data
from src.models.models import build_logistic_model
from src.models.models import build_residual_model
from keras.wrappers.scikit_learn import KerasClassifier


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')


def get_options():
    parser = argparse.ArgumentParser(prog="model data section")
    parser.add_argument('select_model', nargs='+', help='name of the model, select from list in README'),
    parser.add_argument('dataset_path', nargs='+', default=os.path.dirname(os.path.realpath(__file__)).replace("/src","") + '/tmp/dataset.csv', help='path to dataset'),
    parser.add_argument('section', nargs='+', help='name of section in config file'),
    parser.add_argument('--output', default=os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/tmp/" + str(datetime.now()) + '/', help='path to output directory'),
    parser.add_argument('--configs', default=os.path.dirname(os.path.realpath(__file__)) + "/configs.ini", help='path to config file'),
    parser.add_argument('--n_iter', default=6, type=int, help='number of iterations in RandomizedSearchCV'),
    parser.add_argument('--n_jobs', default=1, type=int, help='number of jobs'),
    parser.add_argument('--patience', '-p' , default=100, type=int, help='patience of fit'),
    parser.add_argument('--gridsearch', '-g', action='store_true', default=False, help='use gridsearch'),
    parser.add_argument('--experiments_file', '-e', default='experiments.csv', help='where to write results of experiments')
    parser.add_argument('--length', default=256, help='max length of sequence')
    return parser


def fingerprint(seq, length):
    f = []
    for char in seq:
        if len(f) <= int(length):
            f.append(ord(char))
    while len(f) <= int(length):
        f.append(0)
    return f


def script(args_list, random_state=False, p_rparams=False): 
    scoring = "accuracy"
    time_start = datetime.now()
    if len(sys.argv) > 1:
        options = get_options().parse_args()
    else:
        options = get_options().parse_args(args_list)

    callbacks_list = create_callbacks(options.output, options.patience, options.dataset_path[0])
    
    # writing to a file
    handler = logging.FileHandler(options.output + 'log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # and to stderr (for stdout `stream=sys.stdout`)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    #logging.basicConfig(filename=options.output+'main.log', level=logging.INFO)
    n_folds, epochs, rparams, gparams = read_config(options.configs, options.section[0])
    data = pd.read_csv(options.dataset_path[0])
    data = np.array(data)
    data = data.T
    labels = []
    features = []
    for d in data[1]:
        features.append(fingerprint(d, options.length))
    for l in data[2]:
        labels.append(int(l))
    x_train, x, y_train, y = train_test_split(features, labels, test_size=0.2, random_state=random_state)
    x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.8, random_state=random_state)
    
    if options.gridsearch and not p_rparams:
        logger.info("GRID SEARCH")  
        
        if options.select_model[0] == "logreg":
            model = RandomizedSearchCV(LogisticRegression(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, scoring="accuracy")
        elif options.select_model[0] == "knn":
            model = RandomizedSearchCV(KNeighborsClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10,
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "xgb" and xgb_flag:
            model = RandomizedSearchCV(xgb.XGBClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "svc":
            model = RandomizedSearchCV(SVC(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "rf":
            model = RandomizedSearchCV(RandomForestClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "if":
            model = RandomizedSearchCV(IsolationForest(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "regression":
            search_model = KerasClassifier(build_fn=build_logistic_model, input_dim=input_shape, output_dim=output_shape)
            grid = RandomizedSearchCV(estimator=search_model, param_distributions=gparams, n_jobs=options.n_jobs, cv=n_folds, n_iter=options.n_iter, verbose=10, scoring=scoring, refit='MCC')
            rparams = grid.fit(x_train, y_train)
            model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001),
                                     momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))
            search_model = KerasClassifier(build_fn=model, input_dim=input_shape, output_dim=output_shape)
        elif options.select_model[0] == "residual":
            search_model = KerasClassifier(build_fn=build_residual_model, input_dim=input_shape, output_dim=output_shape)
            grid = RandomizedSearchCV(estimator=search_model, param_distributions=gparams, n_jobs=options.n_jobs, cv=n_folds, n_iter=options.n_iter, verbose=10, scoring=scoring, refit='MCC')
            rparams = grid.fit(x_train, y_train)
            model = build_logistic_model(input_shape, output_shape, activation=rparams.get("activation"),
                                     loss=rparams.get("loss"), metrics=rparams.get("metrics"),
                                     optimizer=rparams.get("optimizer"), learning_rate=rparams.get("learning_rate"),
                                     momentum=rparams.get("momentum"), init_mode=rparams.get("init_mode")) 
        else:
            logger.info("Model name is not found or xgboost import error.")
            return 0, 0

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
        elif options.select_model[0] == "regression":
            model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001),
                                     momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))
        elif options.select_model[0] == "residual":
            model = build_logistic_model(input_shape, output_shape, activation=rparams.get("activation"),
                                     loss=rparams.get("loss"), metrics=rparams.get("metrics"),
                                     optimizer=rparams.get("optimizer"), learning_rate=rparams.get("learning_rate"),
                                     momentum=rparams.get("momentum"), init_mode=rparams.get("init_mode"))
        else:
            logger.info("Model name is not found.")
            return 0, 0

        logger.info("FIT")

        if options.select_model[0] == "regression" or options.select_model[0] == "residual":
            history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size"), epochs=epochs, validation_data=(x_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list)
        else:
            history = model.fit(x_train, y_train)

    if options.gridsearch and not p_rparams:
        rparams = model.cv_results_

    logger.info("EVALUATE")
    train_acc, test_acc, rec = evaluate(logger, options, random_state, options.output, model, x_train, x_test, x_val, y_train, y_test, y_val, time_start, rparams, history, options.length[0], options.section[0], "", n_jobs=options.n_jobs)
    return train_acc, test_acc, rparams, rec
    

if __name__ == "__main__":
    args_list = ['rf', 'tmp/dataset.csv', 'RF_TOX21', '--n_jobs', '-1', '--n_iter', '70', '-p', '2000', '--length', '512']
    script(args_list)
