# COBS
Classification of biochemical sequences

The project goal is to develop a framework for the classification of biochemical sequences using tools from the scikit-learn package and other models like LSTM or Sequence-to-Sequence, which shows a good result on text strings. Working with sequences like fasta will be the subject of study.

## Install with Conda
- Linux
- Python 3.6 or 2.7
- Install https://github.com/DentonJC/virtual_screening
- run env.sh
- Conda (https://www.anaconda.com/download/#linux)
- conda install --file requirements

## Usage


## Example input
### Single experiment:
### Table of experiments:

## Example output

## Results
### Protein classification: plastid vs mitochondrion
Class Distribution:
- plastids 76532 (72%)
- mitochondrions 29660 (28%)

length of sequences: 256

Model | train accuracy | test accuracy
--- | --- | ---
regression  | |
random forest | |
KNN | |
CSV | |

### DNA classification: Promoter Gene Sequences
Class Distribution:
- positive instances: 53 (50%)
- negative instances: 53 (50%)

Model | train accuracy | test accuracy
--- | --- | ---
regression  | |
random forest | |
KNN | |
CSV | |

### DNA classification: Splice-junction Gene Sequences
Class Distribution:
- EI:       767  (25%)
- IE:       768  (25%)
- Neither: 1655  (50%)

Model | train accuracy | test accuracy
--- | --- | ---
regression  | |
random forest | |
KNN | |
CSV | |

## Datasets
Used:
  - ftp://ftp.ncbi.nlm.nih.gov/refseq/release/plastid/
  - ftp://ftp.ncbi.nlm.nih.gov/refseq/release/mitochondrion/
  - https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29
  - https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Promoter+Gene+Sequences%29

Tested:
  - https://www.ncbi.nlm.nih.gov/unigene/?term=human[organism]
  - ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/
  - ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/archaea/
