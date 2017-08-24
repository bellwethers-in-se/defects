from __future__ import print_function, division

import os
import sys

"Add root to path"
root = os.path.join(os.getcwd().split('src')[0], 'src/defects')
if root not in sys.path:
    sys.path.append(root)

"Other imports"
from prediction.model import rf_model
from utils import *
from metrics.abcd import abcd
import numpy as np
import pandas
from tabulate import tabulate
from data.handler import get_all_projects

"Ignore warnings"
import warnings

warnings.filterwarnings("ignore")


def weight_training(test_instance, training_instance):
    head = training_instance.columns
    new_train = training_instance[head[:-1]]
    new_train = (new_train - test_instance[head[:-1]].mean()) \
                / test_instance[head[:-1]].std()
    new_train[head[-1]] = training_instance[head[-1]]
    new_train.dropna(axis=1, inplace=True)
    tgt = new_train.columns
    new_test = (test_instance[tgt[:-1]] - test_instance[tgt[:-1]].mean()) \
               / (test_instance[tgt[:-1]].std())

    new_test[tgt[-1]] = test_instance[tgt[-1]]
    new_test.dropna(axis=1, inplace=True)
    columns = list(set(tgt[:-1]).intersection(new_test.columns[:-1])) \
              + [tgt[-1]]
    return new_train[columns], new_test[columns]


def predict_defects(train, test):
    actual = test[test.columns[-1]].values.tolist()
    actual = [1 if act == "T" else 0 for act in actual]
    predicted, distr = rf_model(train, test)
    return actual, predicted, distr


def bellw_incr(source, target, n_rep=10):
    """
    TNB: Transfer Naive Bayes
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    for src_name, src_path in source.iteritems():
        if src_name == "lucene":
            for n in xrange(1, len(src_path.data)):
                stats_lst = []  # Initialize list to gather stats
                "Reformat source to in a list"
                src_data = [src_path.data[:n]] if not isinstance(src_path.data[:n], list) else src_path.data[:n]
                name = src_data[0].split('/')[-1].split('.csv')[0]  # Get the name of the bellwether proj
                print("Bellwether Version: {}".format(name))
                for tgt_name, tgt_path in target.iteritems():
                    if not src_name == tgt_name:
                        src = list2dataframe(src_data)
                        tgt = list2dataframe([tgt_path.data[-1]])
                        pd, pf, pr, f1, g, auc = [], [], [], [], [], []
                        for _ in xrange(n_rep):
                            _train, __test = weight_training(test_instance=tgt,
                                                             training_instance=src)

                            actual, predicted, distribution = predict_defects(
                                    train=_train, test=__test)

                            p_d, p_f, p_r, rc, f_1, e_d, _g, auroc = \
                                abcd(actual, predicted, distribution)

                            pd.append(p_d)
                            pf.append(p_f)
                            pr.append(p_r)
                            f1.append(f_1)
                            g.append(e_d)
                            auc.append(int(auroc))

                        stats_lst.append([tgt_name, int(np.mean(pd)), int(np.mean(pf)),
                                          int(np.mean(pr)), int(np.mean(f1)),
                                          int(np.mean(g)), int(np.mean(auc))])

                stats = pandas.DataFrame(sorted(stats_lst, key=lambda lst: lst[0], reverse=True),  # Sort by G Score
                                         columns=["Name", "Pd", "Pf", "Prec", "F1", "G", "AUC"])

                print(tabulate(stats
                               , headers=["Name", "Pd", "Pf", "Prec", "F1", "G", "AUC"]
                               , tablefmt="fancy_grid"), end="\n------\n")


def _test_bellw_incr():
    data = get_all_projects()["Apache"]
    bellw_incr(source=data, target=data)
    set_trace()


if __name__ == "__main__":
    _test_bellw_incr()
