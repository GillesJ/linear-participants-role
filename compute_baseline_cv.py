#!/usr/bin/env python3
"""
compute_baseline_cv.py
cbrole 
12/8/17
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
from sklearn.externals.joblib import load
import datahandler as dh
from sklearn.model_selection import StratifiedKFold
import numpy as np
from cbrole import reporter


def print_overview(all_scores):

    cv_scores = {}
    score_keys = all_scores[0].keys()
    for k in score_keys:
        cv_scores[k] = [score[k] for score in all_scores]
        if "_all_labels" in k:
            cv_scores["{}_avg".format(k)] = np.nanmean(
                np.array(cv_scores[k], dtype=np.float), axis=0
            )
        else:
            cv_scores["{}_avg".format(k)] = np.nanmean(
                np.array(cv_scores[k], dtype=np.float)
            )

    for k, v in cv_scores.iteritems():
        if "_avg" in k:
            print("{}: {}".format(k, np.round(v, decimals=4) * 100))


RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=RANDOM_STATE)

dataset = "nl"

data_dir = {
    "en": "/home/gilles/repos/cbrole/static/BASIC_en_3_CBROLE_150236394374",
    "nl": "/home/gilles/repos/cbrole/static/NL_LINEARSVC_nl_3_CBROLE_150705402137",
}
run_dir = data_dir[dataset]

X_in, y_in = dh.load_data("{}/holdin.svm".format(run_dir), memmapped=False)

folds = cv.split(X_in, y_in)

all_scores = []
baseline_majority_scores = []
baseline_random_scores = []
for (i, (train_idc, test_idc)) in enumerate(folds):
    X_test = X_in[test_idc]
    y_test = y_in[test_idc]

    bl_maj = reporter.majority_baseline(y_test)
    baseline_majority_scores.append(bl_maj)
    bl_rand = reporter.random_baseline(y_test)
    baseline_random_scores.append(bl_rand)

print("\nCV majority baseline")
print_overview(baseline_majority_scores)
print("\nCV random baseline")
print_overview(baseline_random_scores)
