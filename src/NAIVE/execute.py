from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src/defects')
if root not in sys.path:
    sys.path.append(root)

from prediction.model import rf_model
from utils import *
from metrics.abcd import abcd
import numpy as np
import pandas
from tabulate import tabulate


import warnings
warnings.filterwarnings("ignore")


def weight_training(test_instance, training_instance):
    head = training_instance.columns
    new_train = training_instance[head[:-1]]
    new_train = (new_train - test_instance[head[:-1]].mean()) / test_instance[head[:-1]].std()
    new_train[head[-1]] = training_instance[head[-1]]
    new_train.dropna(axis=1, inplace=True)
    tgt = new_train.columns
    new_test = (test_instance[tgt[:-1]] - test_instance[tgt[:-1]].mean()) / (
        test_instance[tgt[:-1]].std())

    new_test[tgt[-1]] = test_instance[tgt[-1]]
    new_test.dropna(axis=1, inplace=True)
    columns = list(set(tgt[:-1]).intersection(new_test.columns[:-1])) + [tgt[-1]]
    return new_train[columns], new_test[columns]


def predict_defects(train, test):
    actual = test[test.columns[-1]].values.tolist()
    actual = [1 if act == "T" else 0 for act in actual]
    predicted, distr = rf_model(train, test)
    return actual, predicted, distr


def bellw(source, target, n_rep=12, verbose=False):
    """
    TNB: Transfer Naive Bayes
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    result = dict()
    stats = []
    print("Bellwether")
    for tgt_name, tgt_path in target.iteritems():
        charts = []
        # print("{} \r".format(tgt_name[0].upper() + tgt_name[1:]))
        val = []
        for src_name, src_path in source.iteritems():
            if src_name == "lucene":
                if not src_name == tgt_name:
                    src = list2dataframe([src_path.data[-1]])
                    tgt = list2dataframe([tgt_path.data[-1]])

                    pd, pf, pr, f1, g, auc = [], [], [], [], [], []
                    for _ in xrange(n_rep):
                        _train, __test = weight_training(test_instance=tgt, training_instance=src)
                        # _train, __test = tgt, src
                        actual, predicted, distribution = predict_defects(train=_train, test=__test)
                        p_d, p_f, p_r, rc, f_1, e_d, _g, auroc = abcd(actual, predicted, distribution)

                        pd.append(p_d)
                        pf.append(p_f)
                        pr.append(p_r)
                        f1.append(f_1)
                        g.append(e_d)
                        auc.append(int(auroc))

                    stats.append([tgt_name, int(np.mean(pd)), int(np.mean(pf)),
                                  int(np.mean(pr)), int(np.mean(f1)),
                                  int(np.mean(g)), int(np.mean(auc))])  # ,

    stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[0], reverse=True),  # Sort by G Score
                             columns=["Name", "Pd", "Pf", "Prec", "F1", "G", "AUC"])

    # set_trace()
    print(tabulate(stats
                   , headers=["Name", "Pd", "Pf", "Prec", "F1", "G", "AUC"]
                   , tablefmt="fancy_grid"))

    result.update({tgt_name: stats})

    return result


def bellw_loo(source, target, n_rep=12, verbose=False):
    """
    Naive Learner (with leave-one-ot bellwether test)
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    result = dict()
    for hld_name, hld_path in target.iteritems():
        stats = []
        holdout = hld_name
        print("Holdout: {}".format(holdout))
        for src_name, src_path in source.iteritems():
            if not src_name == holdout:
                pd, pf, pr, f1, g, auc = [], [], [], [], [], []
                for tgt_name, tgt_path in target.iteritems():
                    if src_name != tgt_name and tgt_name != hld_name:
                        src = list2dataframe([src_path.data[-1]])
                        tgt = list2dataframe([tgt_path.data[-1]])
                        for _ in xrange(n_rep):
                            _train, __test = weight_training(test_instance=tgt, training_instance=src)
                            actual, predicted, distribution = predict_defects(train=_train, test=__test)
                            p_d, p_f, p_r, rc, f_1, e_d, _g, auroc = abcd(actual, predicted, distribution)

                            pd.append(p_d)
                            pf.append(p_f)
                            pr.append(p_r)
                            f1.append(f_1)
                            auc.append(int(auroc))
                            g.append(e_d)

                stats.append([src_name, int(np.mean(pd)), int(np.mean(pf)),
                              int(np.mean(pr)), int(np.mean(f1)),
                              int(np.mean(g)), int(np.mean(auc))])  # ,
        stats_df = pandas.DataFrame(sorted(stats, key=lambda lst: lst[-2], reverse=True), columns=["Name", "Pd", "Pf", "Prec", "F1", "AUC", "G"])
        print(stats_df, end="\n-----\n")
        result.update({hld_name: stats_df})
    return stats_df


def bellw_local(source, target, n_rep=12, verbose=False):
    """
    TNB: Transfer Naive Bayes
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    print("Local")
    result = dict()
    stats=[]
    for name, data in target.iteritems():
        train, test = data.data[-2], data.data[-1]
        src = list2dataframe([train])
        tgt = list2dataframe([test])
        pd, pf, pr, f1, g, auc = [], [], [], [], [], []
        for _ in xrange(n_rep):
            actual, predicted, distribution = predict_defects(train=src, test=tgt)
            p_d, p_f, p_r, rc, f_1, e_d, _g, auroc = abcd(actual, predicted, distribution)

            pd.append(p_d)
            pf.append(p_f)
            pr.append(p_r)
            f1.append(f_1)
            g.append(e_d)
            auc.append(int(auroc))

        stats.append([name, int(np.mean(pd)), int(np.mean(pf)),
                      int(np.mean(pr)), int(np.mean(f1)),
                      int(np.mean(g)), int(np.mean(auc))])  # ,

    stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[0], reverse=True),  # Sort by G Score
                             columns=["Name", "Pd", "Pf", "Prec", "F1", "G", "AUC"])

    print(tabulate(stats
                   , headers=["Name", "Pd", "Pf", "Prec", "F1", "G", "AUC"]
                   , tablefmt="fancy_grid"))

    return stats



def bellw_jur():
    from data.handler import get_all_projects
    all = get_all_projects()
    set_trace()
    apache = all["Apache"]
    return bellw(apache, apache, n_rep=10)


if __name__ == "__main__":
    bellw_jur()
