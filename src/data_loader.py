#!/usr/bin/env python
"""

"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed  
import multiprocessing
from Bio import Entrez
from Bio import SeqIO
Entrez.email = "your@mail.com"

data_path = os.path.dirname(os.path.realpath(__file__)).replace("/src","") + '/data'
dataset_path = os.path.dirname(os.path.realpath(__file__)).replace("/src","") + '/tmp'
if not os.path.exists(data_path):
    os.makedirs(data_path)
if not os.path.exists(dataset_path ):
    os.makedirs(dataset_path )


def database(item, label, length):
    features = []
    labels = []    
    search_string = item # + " AND " + animal + "[Organism]"
    handle = Entrez.esearch(db="protein", term=search_string, retmax=length)
    search = Entrez.read(handle)
    for i, s in enumerate(search["IdList"]):
        print("Processing "+str(i+1)+" / "+str(length))
        handle = Entrez.esummary(db="protein", id=s)
        result = Entrez.read(handle)
        handle = Entrez.efetch(db="protein", id=result[0]["Gi"], rettype="gb", retmode="text")
        result = SeqIO.read(handle, "genbank")
        features.append(str(result.seq))
        labels.append(label)

    features = np.array(features).T
    labels = np.array(labels).T
    data = np.c_[features, labels]
    print("Loading dataset complete")
    return data


def local(data_path):
    features = []
    labels = []
    files = os.listdir(data_path)
    label = 0
    for f in files:
        records = list(SeqIO.parse(data_path + '/' + f, "fasta"))
        for r in records:
            features.append(str(r.seq))
            labels.append(label)
        label+=1

    features = np.array(features).T
    labels = np.array(labels).T
    data = np.c_[features, labels]

    print("Creating dataset complete")
    return data
    

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    local(data_path)
    data = []
    data.append(Parallel(n_jobs=num_cores, verbose=5)(delayed(database)(items, label, length) for (items, label, length) in [['rhodopsin', 0, 2500], ['MHC', 1, 2500], ['hemoglobin', 2, 2500]]))
    data = np.c_[data[0][0].T, data[0][1].T]
    data = data.T
    print(data)
    df = pd.DataFrame(data)
    df.to_csv(dataset_path + '/' + "dataset.csv")
