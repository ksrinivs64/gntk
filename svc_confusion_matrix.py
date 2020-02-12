import numpy as np
from os.path import join
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
import argparse


def search(data_dir):
    gram = np.load(join(data_dir, 'gram.npy'))
    gram /= gram.min()

    y = np.load(join(data_dir, 'labels.npy'))

    svc = SVC(C=0.09077853991937558, kernel='precomputed', cache_size=16000, max_iter=5e5)
    y_pred = cross_val_predict(svc, gram, y, cv=StratifiedKFold(n_splits=3))
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)


parser = argparse.ArgumentParser(description='hyper-parameter search')
parser.add_argument('--data_dir', type=str, required=True, help='data_dir')
args = parser.parse_args()
search(args.dataset, args.data_dir)
