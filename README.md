# Modeling and prediction of drug toxicity from chemical structure

NIH Tox21 Challenge: https://tripod.nih.gov/tox21/challenge/

## Pipeline

### Append labels to test and score datasets: `label_score_data.py`, `label_test_data.py`

The test and score datasets provided by the Tox21 Challenge are unlabeled - the labels are provided separately from the SMILES files. To generate labeled SMILES files matching the training dataset, run `label_score_data.py` and `label_test_data.py`.

The data in data_raw has already been labeled accordingly.

### Feature extraction: `smiles_to_features.py`

Read in SMILES codes and labels (active/inactive) provided by the Tox21 Challenge and output featurized data.

Run `python smiles_to_features.py --help` to learn more about the different arguments

Example: `python smiles_to_features.py --assay_name nr-ahr --smiles_dir data_raw --smiles_file_ext smiles --dataset train --features_dir data_features --features_file_ext features --attempts 15 --delay 1`

Modify the relevant_features list in the code to choose feature set. See PubChem and PubChemPy notes below for more information about available compound properties.

To generate a script to run feature extraction over all datasets, run `python feature_retriever_script_generator.py`

### Tune hyperparameters: `train.py`

Train a (deep) neural net model with specified hyperparameters on the train and test (validation) datasets.

Run `python train.py --help` to learn more about the different arguments

Example: `python train.py --run_id 0 --rand_seed 848 --assay_name nr-ahr --res_freq 25 --plot True --tensorboard True --loss_balance True --kernel_reg_const 0 --batch_size 50 --num_epochs 4 --node_array 256 256 256`

To generate a script to test multiple hyperparameters over all datasets, modify `hyperparameter_tuning_script_generator.py` to specify hyperparmeters options and run `python hyperparameter_tuning_script_generator.py`.

### Evaluate a trained model on the score dataset: `score.py`

Evaluate a specific trained model on the score dataset, or find the optimal trained model (from multiple results files) and evaluate that model on the score dataset.

Run `python score.py --help` to learn more about the different arguments

Example (evaluate a specific trained model): `python score.py --run_id 0 --assay_name nr-ahr --results_dir results --results_file_ext results --eval_metric auroc_test --saliency True`

Example (find and evaluate optimal trained model): `python score.py --assay_name nr-ahr --results_dir results --results_file_ext results --eval_metric auroc_test --saliency True`

## Other files

* `models.py`: defines a model class, in this case, a neural network
* `util.py`: utility functions for reading data
* `baseline.ipynb`: Jupyter notebook for evaluting basic models in scikit-learn on the datasets

## Key dependencies

* python 3
* pubchempy 1.0.4
* tensorflow 1.4
* numpy 1.13
* pandas 0.21
* scikit-learn 0.19
* matplotlib 2.1
* jupyter 1.0

## Directory structure

### Data
[data_dir]/[dataset]/[assay_name].[data_file_ext]
* [data_dir]: name of data directory
  * data_raw: the original data files downloaded from Tox21 Challenge
  * data_features: featurized data files
* [dataset]: train, test, or score dataset
  * train: use to train the model
  * test: use to tune hyperparameters
  * score: final evaluation dataset
* [assay_name]: one of 12 different assays used in the Tox21 Challenge
* [data_file_ext]: file extension (excluding the period '.') of the data files

### Results
[results_dir]/[assay_name]/[run_id].[results_file_ext]
* [results_dir]: name of results directory
* [assay_name]: one of 12 different assays used in the Tox21 Challenge
* [run_id]: unique integer id assigned to hyperparameter set
* [results_file_ext]: file extension (excluding the '.') of the results files

If Tensorboard is enabled (via the `--tensorboard` argument for `train.py`), event files are written to [results_dir]/[assay_name]/[run_id]_tb/

Sample results files from running the "Example" commands are included in this repo under results/nr-ahr/.

## Known Issues

* AUROC calculation using Tensorflow. The values are different from values calculated using scikit-learn, which is presumably correct.

## Future work

* Modify model to train and evaluate all assay datasets simultaneously. The input would be  the same (a set of featurized compounds data, shape = (num_examples, num_features)), but the output would become a layer of shape (num_assays,) representing the logit of the  classification probabilities for each of the 12 assays.

## PubChem and PubChemPy Notes

### PubChem
* [PubChem Docs](https://pubchemdocs.ncbi.nlm.nih.gov/)
  * [List of available pre-computed compound properties](https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest$_Toc494865567)
    * [List and description of the 881 PubChem fingerprints](ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt)
    * 14 3-D properties (e.g. 'Volume3D') are only available for small, rigid, organic compounds. See [here](https://pubchem.ncbi.nlm.nih.gov/release3d.html)

### PubChemPy

PubChemPy is a Python wrapper for the PUG-REST service to access PubChem. Note that the default time-out on PubChem servers is 30 seconds (see [here](https://pubchemdocs.ncbi.nlm.nih.gov/programmatic-access)). As a result, retrieving compound properties (i.e. features) may result in `Bad HTTP Requests` and other errors. The `smiles_to_features.py` script attempts to mitigate this issue by trying multiple attempts until successful. However, in certain (unclear) cases, the script becomes stuck in an attempt.

A suggested workaround is as follows:
* Run the script in the background via bash: `python -u smiles_to_features.py --args > tmp.out &`
  * The `-u` argument forces python to flush stdout immediately without buffering. Since `smiles_to_features.py` prints a status for each compound it tries to featurize, this allows you to see realtime status of the script - e.g., whether it has become stuck in an attempt
  * Redirect stdout to a file (i.e., tmp.out) to be able to monitor the status without having the script constantly print to the shell
  * The `&` at the end of the command tells bash to run the script in the background.
* If the script gets stuck (e.g., `tail tmp.out` stops showing progress):
  1. Foreground the job: `fg %[job_id]`
      * Find the `[job_id]` using the `jobs` program
  2. Cancel the current stuck attempt: `Ctrl-C`
  3. Pause the script: `Ctrl-Z`
  4. Background the script and continue running: `bg %[job_id]`

______

Built on a CS 221 project by Joyce Kang, Rifath Rashid, and Benjamin Yeh (myself): https://github.com/RifathRashid/biological-assay-classification