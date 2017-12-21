import argparse
import pandas as pd
import util

def reduce_to_shared_features(filenames):
    ## get set of shared features ##

    # initialize set of shared features
    shared_features = None
    all_features = set()

    for filename in filenames:
        with open(filename, 'r') as data_file:
            # read header
            features = set(data_file.readline().strip().split('\t'))
        all_features |= features
        if shared_features == None:
            shared_features = features
        else:
            shared_features &= features

    print('the following features will be removed: ' + str(all_features-shared_features))
    shared_features = list(shared_features)

    ## remove features not shared across datasets ##

    for filename in filenames:
        # only read shared features from CSV file
        df = pd.read_csv(filename, sep='\t', header=0, float_precision='high', usecols=shared_features)

        # order columns
        col_order = ['cid', 'ncats_id', 'smiles', 'label', 'fingerprint']
        extra_features = [col for col in df.columns if col not in col_order]
        if len(extra_features) > 0:
            extra_features.sort()
            col_order.extend(extra_features)
        df = df[col_order]

        # write out CSV
        df.to_csv(filename, sep='\t', index=False)

def main():
    # construct parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, required=True, help='directory to store train, test, and score features files')
    parser.add_argument('--features_file_ext', type=str, default='features', help='features file extension, exluduing the period (e.g. ''features'')')
    args = parser.parse_args()
    
    # parse arguments
    features_dir = args.features_dir
    features_file_ext = args.features_file_ext

    assay_names = ['nr-ahr', 'nr-ar-lbd', 'nr-aromatase', 'nr-ar', 'nr-er-lbd', 'nr-er',
        'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53']

    for assay_name in assay_names:
        filenames = util.get_data_filenames(features_dir, features_file_ext, assay_name)
        filenames = [filenames[key] for key in filenames]
        reduce_to_shared_features(filenames)

if __name__ == "__main__":
    main()