import pubchempy as pcp
import pandas as pd
import os, argparse, errno
# import util

def get_compound_from_smiles(smiles, attempts=5, delay=1):
    '''
    Returns a pubchempy.Compound object matching the smiles code

    Parameters
    - smiles: str
        SMILES code
    - attempts: int
        Number of attempts to try to retrieve PubChem record of compound.
        Attempts may fail due to lack of a matching PubChem record, connection timing out, etc.
    - delay: float
        Seconds to wait between each attempt

    Returns
    - compound: pubchempy.Compound
        First pubchempy.Compound object with a valid cid that matches the input smiles code
        Returns None if fails to get compound from smiles for any reason
    '''

    while attempts >= 1:
        try:
            # get list of all Compound objects matching the provided smiles code
            compounds = pcp.get_compounds(smiles, namespace='smiles')

            # choose the first matching Compound object with a valid cid (PubChem Compound ID)
            for compound in compounds:
                if compound.cid != None:
                    return compound
        except:
            attempts -= 1
            print('Could not get compound ' + smiles + '. ' + str(attempts) + ' attempts remaining.')
            time.sleep(delay)
        else:
            # From https://pubchempy.readthedocs.io/en/latest/guide/gettingstarted.html:
            # 'line notation inputs like SMILES and InChI can return automatically generated records
            # that aren’t actually present in PubChem, and therefore have no CID'
            print('No PubChem record for ' + smiles)
            return None
    
    if attempts <= 0:
        print('Failed to get compound from smiles after exhausting all attempts')
        return None

def smiles_to_DataFrame(smiles_filename, attempts, delay):
    '''
    Gets properties of all compounds in smiles_file and arranges them in a pandas DataFrame.

    Parameters
    - smiles_filename: str
        Filename of smiles file (e.g. provided by Tox21 Challenge).
        Data must be arranged as smiles, ncats_id, label with no header
    
    Return
    - df: pandas.DataFrame
        pandas DataFrame where rows are different compounds and columns are attributes
        of the pubchempy.Compound class. Note that some compounds may have missing data.
    '''

    # list of pandas Series objects for each compound
    compound_series_list = []

    with open(smiles_filename, 'r') as smiles_file:
        for index, line in enumerate(smiles_file):
            # parse line
            parsed_line = line.strip().split('\t')
            smiles, ncats_id, label = tuple(parsed_line)
            
            # get compound
            print('Getting compound ' + str(index))
            compound = get_compound_from_smiles(smiles, attempts, delay)
            
            # check for empty compound (e.g. failed to get compound from smiles code)
            if compound == None:
                continue
            
            # convert compound to pandas Series
            compound_series = compound.to_series()
            
            # append label, smiles, and ncats_id to pandas Series
            compound_series['label'] = label
            compound_series['smiles'] = smiles
            compound_series['ncats_id'] = ncats_id
            
            # add Series to list of compound Series
            compound_series_list.append(compound_series)

    df = pd.concat(compound_series_list, axis=1)
    df = df.T
    return df

def process_data(df, relevant_features='', rename_features={}):
    '''
    Processes a pandas DataFrame so that it is ready to be used in a learning model
    - all data points are unique and have the same number of features
        Note: since this function is applied separately to train, test, and score datasets,
        they may have slightly different features --> needs to be reconciled at training time
    - order of features: cid, ncats_id, smiles code, label, fingerprint, extra features (sorted)

    Parameters
    - df: pandas.DataFrame
        pandas DataFrame where rows are different compounds and columns are attributes
        of the pubchempy.Compound class. Note that some compounds may have missing data.
    - relevant_features: set of str, optional (default = '')
        - '' (default): keep all extra features
        - None: discard all extra features
        - set of str: keep only specified extra features
    - rename_features: dict[old_name] --> new_name, optional (default = {})
        Dictionary specifying columns to be renamed

    Returns
    - df: pandas.DataFrame
        Processed DataFrame
        - drop features with missing data
        - remove duplicates
        - drop irrelevant features
        - rename features (columns)
        - order features
    '''

    all_cols = [col for col in df.columns]
    col_order = ['cid', 'ncats_id', 'smiles', 'label', 'fingerprint']
    
    # drop columns with any missing data
    df.dropna(axis=1, how='any', inplace=True)
    cols_missing_data = set(all_cols) - set([col for col in df.columns])
    
    # remove duplicates
    df.drop_duplicates(subset='smiles', keep='first', inplace=True)
    df.drop_duplicates(subset='cid', keep='first', inplace=True)
    
    # drop irrelevant features
    if relevant_features == None:
        relevant_features = set()
    if relevant_features != '':
        # remove hex-encoded 'fingerprint', rename 'cactvs_fingerprint' to 'fingerprint'
        if 'cactvs_fingerprint' in df.columns and 'fingerprint' in df.columns:
            df.drop('fingerprint', axis=1, inplace=True)
            df.rename(index=str, columns={'cactvs_fingerprint': 'fingerprint'}, inplace=True)

        relevant_features |= set(col_order)
        irrelevant_cols = [col for col in df.columns if col not in relevant_features]
        df.drop(irrelevant_cols, axis=1, inplace=True)
    
        # print out relevant features that were dropped
        relevant_dropped_nonexist = [col for col in relevant_features if col not in all_cols]
        relevant_dropped_missing_data = [col for col in relevant_features if col in cols_missing_data]
        print('relevant features not found in the dataset: %s' % str(relevant_dropped_nonexist))
        print('relevant features dropped to due missing data: %s' % str(relevant_dropped_missing_data))
    
    # rename column names, e.g. cactvs_fingerprint to fingerprint
    df.rename(index=str, columns=rename_features, inplace=True)

    # order columns as 'cid', 'ncats_id', 'smiles', 'label', 'fingerprint', extra features (sorted)
    extra_features = [col for col in df.columns if col not in col_order]
    if len(extra_features) > 0:
        extra_features.sort()
        col_order.extend(extra_features)
    df = df[col_order]
    return df

def main():
    # construct parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--assay_name', type=str, required=True, help='assay name, e.g. nr-ar, sr-are, ...')
    parser.add_argument('--smiles_dir', type=str, required=True, help='directory to find train, test, and score smiles files')
    parser.add_argument('--smiles_file_ext', type=str, default='smiles', help='smiles file extension, exluduing the period (e.g. ''smiles'')')
    parser.add_argument('--dataset', type=str, required=True, help='which dataset to process: train, test, or score')
    parser.add_argument('--features_dir', type=str, required=True, help='directory to store train, test, and score features files')
    parser.add_argument('--features_file_ext', type=str, default='features', help='features file extension, exluduing the period (e.g. ''features'')')
    parser.add_argument('--attempts', type=int, default=5, help='number of attempts to try to retrieve PubChem record of compound')
    parser.add_argument('--delay', type=float, default=1.0, help='seconds to wait between each attempt at retrieving the PubChem record of a compound')
    args = parser.parse_args()
    
    # parse arguments
    assay_name = args.assay_name
    smiles_dir = args.smiles_dir
    smiles_file_ext = args.smiles_file_ext
    dataset = args.dataset
    features_dir = args.features_dir
    features_file_ext = args.features_file_ext
    attempts = args.attempts
    delay = args.delay

    # get filenames
    # filenames = util.get_data_filenames(smiles_dir, smiles_file_ext, assay_name)
    # smiles_filename = filenames[dataset]
    smiles_filename = os.path.join(os.getcwd(), smiles_dir, dataset, '') + assay_name + '.' + smiles_file_ext
    features_dir_full = os.path.join(os.getcwd(), features_dir, dataset, '')
    features_filename = features_dir_full + assay_name + '.' + features_file_ext

    # make features (output) directory if it does not exist
    if not os.path.exists(features_dir_full):
        try:
            os.makedirs(features_dir_full)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # other arguments
    # rename_features = {'catvs_fingerprint': 'fingerprint'}
    relevant_features = set(['atom_stereo_count', 'bond_stereo_count', 'charge', 'complexity',
        'covalent_unit_count', 'defined_atom_stereo_count', 'defined_bond_stereo_count',
        'exact_mass', 'cactvs_fingerprint', 'h_bond_acceptor_count', 'h_bond_donor_count',
        'heavy_atom_count', 'isotope_atom_count', 'molecular_weight', 'monoisotopic_mass',
        'rotatable_bond_count', 'tpsa', 'undefined_atom_stereo_count',
        'undefined_bond_stereo_count','xlogp'])

    # run
    df = smiles_to_DataFrame(smiles_filename, attempts, delay)
    df = process_data(df, relevant_features)
    df.to_csv(features_filename, sep='\t', index=False)

if __name__ == "__main__":
    main()