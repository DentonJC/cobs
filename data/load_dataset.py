#!/usr/bin/env python
"""

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
    parser = argparse.ArgumentParser(prog="model data")
    parser.add_argument('dataset_path', nargs='?', default=os.path.dirname(os.path.realpath(__file__)).replace("/src",""), help='path to dataset'),
    parser.add_argument('search_path', nargs='?', default=os.path.dirname(os.path.realpath(__file__)).replace("/src","") + '/search.ini', help='path to search config'),
    parser.add_argument('emai', default="your@mail.com", help='your email')
    return parser


def read_config(search_path):
    config.read(search_path)
    def_config = config['DEFAULT']
    SEARCH = eval(def_config['SEARCH'])
    return SEARCH


def database(item, label, length):
    features = []
    labels = []    
    search_string = item # + " AND " + animal + "[Organism]"
    handle = Entrez.esearch(db="protein", term=search_string, retmax=length)
    search = Entrez.read(handle)
    for i, s in enumerate(search["IdList"]):
        try:
            print("Processing " + str(item) + ": " + str(i + 1) + " / " + str(length))
            handle = Entrez.esummary(db="protein", id=s)
            result = Entrez.read(handle)
            handle = Entrez.efetch(db="protein", id=result[0]["Gi"], rettype="gb", retmode="text")
            result = SeqIO.read(handle, "genbank")
            features.append(str(result.seq))
            labels.append(label)
        except:
            break

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
    
    num_cores = multiprocessing.cpu_count()
    data = []
    search = read_config(options.search_path)
    print(search)
    return 0
    data.append(Parallel(n_jobs=num_cores, verbose=5)(delayed(database)(items, label, length) for (items, label, length) in search))
    data = np.c_[data[0][0].T, data[0][1].T]
    data = data.T
    df = pd.DataFrame(data)
    df.to_csv(dataset_path + '/' + "dataset.csv")

if __name__ == "__main__":
    main()
