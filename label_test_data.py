def label_test_data(smiles_filename, sdf_filename, labeled_filename, assay_name):
    '''
    Label test data

    Arguments
    - smiles_filename: str
        Unlabeled test dataset provided by the Tox21 Challenge.
        Each line is smiles code + \t + ncats_id
    - sdf_filename: str
        Labeled test data provided by the Tox21 Challenge as an SDF file
    - labeled_filename: str
        Output file in the format smiles \t ncats_id \t label, matching the format of the training data
    - assay_name: str
        Assay name
    '''
    with open(smiles_filename, 'r') as smiles_file:
        with open(sdf_filename, 'r') as sdf_file:
            with open(labeled_filename, 'w') as labeled_file:
                for line in smiles_file:
                    # ignore comments
                    if line[0] == '#':
                        continue
                    
                    # Get compound ID
                    parsed_line = line.split('\t')
                    ncats_id = parsed_line[1].strip()
                    
                    # Read through SDF file until finding the compound
                    while True:
                        temp = sdf_file.readline().strip()
                        if temp == ncats_id:
                            break
                    
                    # Within the section of the compound, look for labels for the assay
                    while True:
                        temp = sdf_file.readline().strip().lower()
                        if temp == '$$$$': # reached end of section of the compound
                            break
                        elif '<' + assay_name + '>' in temp:
                            labeled_file.write(line.strip())
                            label = sdf_file.readline().strip()
                            labeled_file.write('\t' + label + '\n')

# list of all assays
assay_names = ['nr-ahr', 'nr-ar-lbd', 'nr-aromatase', 'nr-ar', 'nr-er-lbd', 'nr-er', 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53']

# test data - only used to evaluate performance for the leaderboard
smiles_filename = './data_raw/test/tox21_10k_challenge_test.smiles'
sdf_filename = './data_raw/test/tox21_10k_challenge_test.sdf'
for assay in assay_names:
    labeled_filename = './data/test/' + assay + '.smiles'
    label_test_data(smiles_filename, sdf_filename, labeled_filename, assay)

