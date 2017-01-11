from __future__ import print_function, division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src/defects')
if root not in sys.path:
    sys.path.append(root)

import warnings
from prediction.model import nbayes, rf_model
from py_weka.classifier import classify
from utils import *
from metrics.abcd import abcd
from metrics.recall_vs_loc import get_curve
from pdb import set_trace
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas
from plot.effort_plot import effort_plot
from tabulate import tabulate

def weight_training(test_instance, training_instance):
    head = training_instance.columns
    new_train = training_instance[head[:-1]]
    new_train = (new_train - test_instance[head[:-1]].min()) / (test_instance[head[:-1]].max() - test_instance[head[:-1]].min())
    new_train[head[-1]] = training_instance[head[-1]]
    return new_train


def predict_defects(train, test):

    actual = test[test.columns[-1]].values.tolist()
    actual = [1 if act == "T" else 0 for act in actual]
    predicted, distr = rf_model(train, test)
    return actual, predicted, distr


def bellw(source, target, n_rep=12):
    """
    TNB: Transfer Naive Bayes
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    result = dict()
    for tgt_name, tgt_path in target.iteritems():
        stats = []
        charts = []
        print("{} \r".format(tgt_name[0].upper() + tgt_name[1:]))
        val = []
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:

                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)

                pd, pf, g, auc = [], [], [], []
                for _ in xrange(n_rep):
                    _train = weight_training(test_instance=tgt, training_instance=src)
                    __test = (tgt[tgt.columns[:-1]] - tgt[tgt.columns[:-1]].min()) / (
                        tgt[tgt.columns[:-1]].max() - tgt[tgt.columns[:-1]].min())
                    __test[tgt.columns[-1]] = tgt[tgt.columns[-1]]
                    actual, predicted, distribution = predict_defects(train=_train, test=__test)
                    # loc = tgt["$loc"].values
                    # loc = loc * 100 / np.max(loc)
                    # recall, loc, au_roc = get_curve(loc, actual, predicted, distribution)
                    # effort_plot(recall, loc,
                    #             save_dest=os.path.abspath(os.path.join(root, "plot", "plots", tgt_name)),
                    #             save_name=src_name)
                    p_d, p_f, p_r, rc, f_1, e_d, _g, auroc = abcd(actual, predicted, distribution)

                    pd.append(p_d)
                    pf.append(p_f)
                    g.append(_g)
                    auc.append(int(auroc))
                stats.append([src_name, int(np.mean(pd)), int(np.std(pd)),
                              int(np.mean(pf)), int(np.std(pf)),
                              int(np.mean(auc)), int(np.std(auc))])  # ,

        stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[-2], reverse=True),  # Sort by G Score
                                 columns=["Name", "Pd (Mean)", "Pd (Std)",
                                          "Pf (Mean)", "Pf (Std)",
                                          "AUC (Mean)", "AUC (Std)"])  # ,
        # "G (Mean)", "G (Std)"])
        print(tabulate(stats,
                       headers=["Name", "Pd (Mean)", "Pd (Std)",
                                "Pf (Mean)", "Pf (Std)",
                                "AUC (Mean)", "AUC (Std)"],
                       showindex="never",
                       tablefmt="fancy_grid"))

        result.update({tgt_name: stats})
    return result


def tnb_jur():
    from data.handler import get_all_projects
    all = get_all_projects()
    # set_trace()
    apache = all["Apache"]
    return bellw(apache, apache, n_rep=10)


if __name__ == "__main__":
    tnb_jur()
