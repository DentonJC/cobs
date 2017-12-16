#!/usr/bin/env python
"""

"""
import os
import numpy as np
import pandas as pd
from Bio import SeqIO

data_path = os.path.dirname(os.path.realpath(__file__)).replace("/src","") + '/data'
dataset_path = os.path.dirname(os.path.realpath(__file__)).replace("/src","") + '/tmp'
if not os.path.exists(data_path):
    os.makedirs(data_path)
if not os.path.exists(dataset_path ):
    os.makedirs(dataset_path )

def create_dataset():
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

    df = pd.DataFrame(data)
    df.to_csv(dataset_path + '/' + "dataset.csv")
    return data
    

if __name__ == "__main__":
    create_dataset()
