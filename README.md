# COBS
Classification of biochemical sequences

The project goal is to develop a framework for the classification of biochemical sequences using tools from the scikit-learn package and other models like LSTM or Sequence-to-Sequence, which shows a good result on text strings. Working with sequences like fasta will be the subject of study.

Models available:
- KNN (knn)
- Logistic regression (logreg)
- RandomForestClassifier (rf)
- SVC (svc)
- Isolation Forest (if)
- ResidualNN (residual)
- Perceptron (perceptron)
- Multilayer perceptron (m_perceptron)

Models in progress:
- LSTM
- RNN

Use cobs/config.ini to configure the models.
- rparams for basic configuration
- gparams for randomized search configuration

KNOWN BUG in Parallel: need to restart script after using keras model in experiments table.

## Table of Contents
1. [Install](#install)
2. [Usage](#usage)
3. [Input](#input)
4. [Output](#output)
6. [Datasets](#datasets)
5. [Results](#results)
6. [Resources](#resources)

## Install with Conda <a name="install"></a>
- Linux
- Python 3.6 or 2.7
- Install https://github.com/DentonJC/virtual_screening
- run env.sh
- Conda (https://www.anaconda.com/download/#linux)
- conda install --file requirements

## Usage <a name="usage"></a>

    usage: select_model dataset_path [-h] [--output OUTPUT] [--configs CONFIGS] [--n_iter N_ITER]
                  [--n_jobs N_JOBS] [--patience PATIENCE] [--gridsearch]
                  [--experiments_file EXPERIMENTS_FILE] [--length LENGTH]
                  select_model [select_model ...] dataset_path
                  [dataset_path ...]

    positional arguments:
    select_model          name of the model, select from list in README
    dataset_path          path to dataset

    optional arguments:
    -h, --help            show this help message and exit
    --output OUTPUT       path to output directory
    --configs CONFIGS     path to config file
    --n_iter N_ITER       number of iterations in RandomizedSearchCV
    --n_jobs N_JOBS       number of jobs
    --patience PATIENCE, -p PATIENCE    patience of fit
    --gridsearch, -g      use gridsearch
    --experiments_file EXPERIMENTS_FILE, -e EXPERIMENTS_FILE address where to write results of experiments
    --length LENGTH, -l LENGTH    maximum length of sequences
    --targets TARGETS, -t TARGETS    set number of target column

## Example input <a name="input"></a>
#### Single experiment:
python cobs/run_model.py logreg data/dataset.csv --n_jobs -1 --n_iter 6 -p 2000 --length 256 -g
#### Table of experiments:
1. Fill in the table with experiments parameters (examples in /etc, False = empty cell)
2. Run python run.py
3. Experiments will be performed one by one and fill in the columns with the results

## Example output <a name="output"></a>
2018-01-05 19:55:57,028 [__main__] INFO: GRID SEARCH <br />
2018-01-05 19:55:57,028 [__main__] INFO: FIT <br />
Fitting 10 folds for each of 6 candidates, totalling 60 fits <br />
...  <br />
[Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:  5.7min finished  <br />
2018-01-05 20:01:53,124 [__main__] INFO: Accuracy test: 86.59%  <br />
2018-01-05 20:01:54,589 [__main__] INFO: 0:06:07.959393  <br />
Can't create history plot for this type of experiment  <br />
Report complete, you can see it in the results folder  <br />
2018-01-05 20:01:54,720 [__main__] INFO: Done  <br />
2018-01-05 20:01:54,720 [__main__] INFO: Results path: /cobs/tmp/2018-01-05  19:55:46.630191/  <br />

## Datasets <a name="datasets"></a>
##### Generate dataset from local files
1. Put FASTA files into data/ folder
2. Run data/create_dataset.py

##### Download dataset from ncbi server
1. Configure search.ini: select requests and name of labels
2. Run data/load_dataset.py

## Results <a name="results"></a>
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
regression  | 100 | 86.04
random forest | 100 | 88.37
KNN | 100 | 81.39
CSV | 100 | 79.06

### DNA classification: Splice-junction Gene Sequences
Class Distribution:
- EI:       767  (25%)
- IE:       768  (25%)
- Neither: 1655  (50%)

Model | train accuracy | test accuracy
--- | --- | ---
regression  | 89.91 | 87.85
random forest | 99.89 | 92.55
KNN | 100 | 88.55
CSV | 100 | 86.12

## Resources <a name="resources"></a>
Used:
  - ftp://ftp.ncbi.nlm.nih.gov/refseq/release/plastid/
  - ftp://ftp.ncbi.nlm.nih.gov/refseq/release/mitochondrion/
  - https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29
  - https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Promoter+Gene+Sequences%29

Tested:
  - https://www.ncbi.nlm.nih.gov/unigene/?term=human[organism]
  - ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/
  - ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/archaea/
