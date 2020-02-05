import numpy as np
from os.path import join
import argparse
import pandas as pd


def dump_data(data_dir, out_file):
    gram = np.load(join(data_dir, 'gram.npy'))
    gram /= gram.min()
    labels = np.load(join(data_dir, 'labels.npy'))

    gram = np.c_[gram, labels]
    df = pd.DataFrame(data=gram)
    df.to_csv(out_file, index=False)


parser = argparse.ArgumentParser(description='dump embeddings to CSV')
parser.add_argument('--data_dir', type=str, required=True, help='data_dir')
parser.add_argument('--out_dir', type=str, required=True, help='out')
args = parser.parse_args()
dump_data(args.data_dir, args.out_dir)
