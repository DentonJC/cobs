#!/usr/bin/env python
"""

"""
import os
import multiprocessing
import numpy as np
import pandas as pd
from joblib import Parallel, delayed  
from Bio import Entrez
from Bio import SeqIO


SEARCH = [['rhodopsin', 0, 10], ['MHC', 1, 10], ['hemoglobin', 2, 10]]
Entrez.email = "your@mail.com"
DATASET_PATH = os.path.dirname(os.path.realpath(__file__)).replace("/src","")


if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)


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


if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    data = []
    data.append(Parallel(n_jobs=num_cores, verbose=5)(delayed(database)(items, label, length) for (items, label, length) in SEARCH)
    data = np.c_[data[0][0].T, data[0][1].T]
    data = data.T
    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH + '/' + "dataset.csv")
