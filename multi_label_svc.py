import numpy as np
import scipy
from multiprocessing import Pool
from os.path import join
from sklearn.svm import SVC
from hyperopt import tpe, Trials,fmin,STATUS_OK
from hyperopt import hp
from sklearn.model_selection import StratifiedKFold, cross_val_score
from util import load_data
import sklearn
import argparse


### classification type
def hyperopt_train_test(params):
    p = {}
    for key in params:
        if key != 'X' and key != 'y' and key != 'cv':
            p[key] = params[key]
    svc = SVC(**p)
    return cross_val_score(svc, params['X'], params['y'], cv=params['cv']).mean()


def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}


def search(dataset, data_dir):
    gram = np.load(join(data_dir, 'gram.npy'))
    gram /= gram.min()
    # normalize the matrix
    gram_diag = np.sqrt(np.diag(gram))
    gram /= gram_diag[:, None]
    gram /= gram_diag[None, :]

    labels = np.load(join(data_dir, 'labels.npy'))

    space4svc = {
        'C':  hp.uniform('C', 0, 4.0),
        'kernel': 'precomputed',
        'cache_size': 16000,
        'cv': StratifiedKFold(n_splits=10, shuffle=True),
        'max_iter':5e5
    }

    trials = Trials()
    space4svc['X'] = gram
    space4svc['y'] = labels
    best = fmin(f, space4svc, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)

    print('best:')
    print(best)
    print('trials:')
    for trial in trials.trials[:2]:
        print(trial)


parser = argparse.ArgumentParser(description='hyper-parameter search')
parser.add_argument('--data_dir', type=str, required=True, help='data_dir')
parser.add_argument('--dataset', type=str, required=True, help='dataset')
args = parser.parse_args()
search(args.dataset, args.data_dir)
