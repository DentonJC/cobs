#!/usr/bin/env python
"""

"""
import os
import numpy as np
import pandas as pd
from Bio import SeqIO


DATA_PATH = os.path.dirname(os.path.realpath(__file__)).replace("/src","")
DATASET_PATH = os.path.dirname(os.path.realpath(__file__)).replace("/src","")


if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)


def local():
    features = []
    labels = []
    files = []
    all_files = os.listdir(DATA_PATH)

    for f in all_files:
        if ".faa" in f: files.append(f)
        if ".fasta" in f: files.append(f)
        if ".fna" in f: files.append(f)

    label = 0
    for f in files:
        records = list(SeqIO.parse(DATA_PATH + '/' + f, "fasta"))
        for r in records:
            features.append(str(r.seq))            
            labels.append(label)
        label += 1

    features = np.array(features).T
    labels = np.array(labels).T
    data = np.c_[features, labels]

    print("Creating dataset complete")
    return data
    

if __name__ == "__main__":
    data = local()
    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH + '/' + "dataset.csv")
