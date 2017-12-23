import pandas as pd

# list of all assays
assay_names = ['nr-ahr', 'nr-ar-lbd', 'nr-aromatase', 'nr-ar', 'nr-er-lbd', 'nr-er', 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53']

# score data - final evaluation dataset used for judging
smiles_filename = './data_raw/score/tox21_10k_challenge_score.smiles'
results_filename = './data_raw/score/tox21_10k_challenge_score.txt'

# read results file into pandas Dataframe
results = pd.read_csv(results_filename, sep='\t', header=0)
results.columns = [name.lower() for name in results.columns] # make all column names (assay names) lowercase

with open(smiles_filename, 'r') as smiles_file:
    for line in smiles_file:
        # ignore comments
        if line[0] == '#':
            continue

        # Get compound ncats_id
        parsed_line = line.strip().split('\t')
        ncats_id = parsed_line[1].strip()

        # line number of compound with ncats_id
        index = results.index[results['sample id'] == ncats_id].tolist()[0]

        for assay in assay_names:
            label = results[assay][index]

            # only add compounds with a label (compounds without valid label data are labeld 'x')
            if label in ['0','1']:
                # append smiles, ncats_id, and label to new file, matching the format of the training data
                labeled_filename = './data_raw/score/' + assay + '.smiles'
                with open(labeled_filename, 'a') as labeled_file:
                    labeled_file.write(line.strip() + '\t' + label + '\n')
