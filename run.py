#!/usr/bin/env python

"""
An additional script to run a series of experiments described in table like etc/experiments.csv
where columns are hyperparameters and rows are experiments.
"""
import os
import math
import random
import pandas as pd
from cobs.run_model import run


def isnan(x):
    """ Checks if the variable x is empty (NaN and float). """
    return isinstance(x, float) and math.isnan(x)


def main(experiments_file, common_gridsearch, random_state, n_cols):
    """
    Check the rows of experiments_file in a loop. If there are no results in the row (empty fields after len(cols)), 
    it takes all values in this row and calls the experiment function until all result fields are filled with step len(result_cols).
    
    Params
    ------
    experiments_file: string
       path to experiments table
    n_cols: int
        number of result columns in table
    common_gridsearch: bool
        one gridsearch for all experiments in row
    random_state: int
        random state of all   
    """
    if not random_state:
        random_state = random.randint(1, 1000)

    table = pd.read_csv(experiments_file)
    keys = ["--n_jobs ", "-p ", "-g ", "--n_iter ", "--length ", "--n_folds "]
    params = ["n_jobs", "Patience", "Gridsearch", "n_iter", "Length", "Split"]
    pos_params = ['Model', 'Data']
    for i in range(table.shape[0]):
        rparams = False
        command = ""
        for p in pos_params:
            command += str(table[p][i]) + " "

        for c, p in enumerate(params):
            if not isnan(table[p][i]):
                if keys[c] in ["-g "]:
                    command += keys[c] + " "
                else:
                    command += keys[c] + str(table[p][i]) + " "

        command = command + "-e " + experiments_file

        for j in range(int(((table.shape[1] - len(params) - len(pos_params)) / n_cols))):
            accuracy_test = accuracy_train = rec = auc = f1 = '-'
            command = command + " -t " + str(j)
            command = command.split()
            print(command)
            if isnan(table.iloc[i, j*n_cols + len(params) + len(pos_params)]):
                if common_gridsearch:
                    accuracy_test, accuracy_train, rec, auc, f1, rparams = run(command, random_state, rparams)
                else:
                    accuracy_test, accuracy_train, rec, auc, f1, rparams = run(command, random_state, False)

                table = pd.read_csv(experiments_file)
                table.iloc[i, j*n_cols+len(params) + len(pos_params)] = accuracy_test
                table.iloc[i, j*n_cols+1+len(params) + len(pos_params)] = accuracy_train
                table.to_csv(experiments_file, index=False)


if __name__ == "__main__":
    common_gridsearch = False
    random_state = 13
    n_cols = 2

    def_experiments_file = 'etc/experiments.csv'
    experiments_file = input('Enter the experiment table address (default is ' + def_experiments_file + '): ')
    if experiments_file == '':
        experiments_file = def_experiments_file

    main(experiments_file, common_gridsearch, random_state, n_cols)
