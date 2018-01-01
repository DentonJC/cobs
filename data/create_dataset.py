#!/usr/bin/env python
"""

"""
import os
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO


def get_options():
    parser = argparse.ArgumentParser(prog="model data")
    parser.add_argument('dataset_path', nargs='?', default=os.path.dirname(os.path.realpath(__file__)).replace("/src",""), help='path to dataset'),
    parser.add_argument('data_path', nargs='?', default=os.path.dirname(os.path.realpath(__file__)).replace("/src",""), help='path to search config'),
    return parser


def local(data_path):
    features = []
    labels = []
    files = []
    all_files = os.listdir(data_path)

    for f in all_files:
        if ".faa" in f: files.append(f)
        if ".fasta" in f: files.append(f)
        if ".fna" in f: files.append(f)

    label = 0
    for f in files:
        records = list(SeqIO.parse(data_path + '/' + f, "fasta"))
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
    options = get_options().parse_args()
    
    if not os.path.exists(options.data_path):
        os.makedirs(options.data_path)
    if not os.path.exists(options.dataset_path):
        os.makedirs(options.dataset_path)
    data = local(options.data_path)
    df = pd.DataFrame(data)
    df.to_csv(options.dataset_path + '/' + "dataset.csv")
