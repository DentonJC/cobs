#!/usr/bin/env python

"""
Loading dataset by processing several Entrez searchs in one table, number of search is a label.
"""

import os
import argparse
import configparser
import multiprocessing
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio import Entrez
from Bio import SeqIO
config = configparser.ConfigParser()


def get_options():
    parser = argparse.ArgumentParser(prog="Creating dataset from Entrez database")
    default_path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "")
    parser.add_argument('dataset_path', type=str, nargs='?', default=default_path, help='path to dataset'),
    parser.add_argument('search_path', type=str, nargs='?', default=default_path + '/search.ini', help='path to search config'),
    parser.add_argument('--email', type=str, default="your@mail.com", help='your email')
    parser.add_argument('--n_jobs', type=int, default=-1, help='number of workers')
    return parser


def database(item, label, length, db="protein"):
    """
    Returns an array of data (characteristics, labels) that can be written to a file.

    Params
    ------
    item: string
        search string
    label: int
        label of item
    length: int
        number of required items
    db: string
        database of search
    Output
    ------
    data: np.array
        concatenation of found data and input labels
    """
    features = []
    labels = []
    handle = Entrez.esearch(db=db, term=item, retmax=length)
    search = Entrez.read(handle)
    for i, s in enumerate(search["IdList"]):
        print("Processing " + str(item) + ": " + str(i + 1) + " / " + str(length))
        handle = Entrez.esummary(db=db, id=s)
        result = Entrez.read(handle)
        handle = Entrez.efetch(db=db, id=result[0]["Gi"], rettype="gb", retmode="text")
        result = SeqIO.read(handle, "genbank")
        features.append(str(result.seq))
        labels.append(label)

    features = np.array(features).T
    labels = np.array(labels).T
    data = np.c_[features, labels]
    print("Loading " + str(item) + " complete")
    return data


def main():
    options = get_options().parse_args()
    Entrez.email = options.email

    if not os.path.exists(options.dataset_path):
        os.makedirs(options.dataset_path)

    data = []
    config.read(options.search_path)
    def_config = config['DEFAULT']
    search = eval(def_config['SEARCH'])
    print(search)
    data.append(Parallel(n_jobs=options.n_jobs, verbose=5)(delayed(database)(items, label, length) for (items, label, length) in search))
    data = np.c_[data[0][0].T, data[0][1].T]
    data = data.T
    df = pd.DataFrame(data)
    df.to_csv(dataset_path + '/' + "dataset.csv")


if __name__ == "__main__":
    main()
