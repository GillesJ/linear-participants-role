#!/usr/bin/env python3
"""
Script to compute the detection (first-stage score) of the cascading classifier

compute_detect_metrics.py
cbrole 
10/21/19
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import numpy as np
import json
import glob
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)
import settings as s


def round_dict_vals(d):
    return {k: round(v * 100, 2) for k, v in d.iteritems()}


language = "nl"

cascade_pipline_y_fps = {
    "en": "/home/gilles/repos/cbrole/static/en_CASCADE_3CBROLE_150894970996/cv_pipelines/maxabsscaler+cascadingclassifier/y_out_true-y_out_pred.json",
    "nl": "/home/gilles/repos/cbrole/static/nl_VOTING_3CBROLE_150999028765/cv_pipelines/f_classif+maxabsscaler+cascadingclassifier/y_out_true-y_out_pred.json",  # this folder is in fact the Dutch cascade (naming error)
}
y_fp = cascade_pipline_y_fps[language]

with open(y_fp, "rt") as y_in:
    ys = json.load(y_in)
y_true = np.array(ys["y_out_true"])
y_pred = np.array(ys["y_out_pred"])
y_true_firststage = np.where(y_true > -1, 1, y_true)
y_pred_firststage = np.where(y_pred > -1, 1, y_pred)

scores = {}
(
    scores["precision"],
    scores["recall"],
    scores["fscore"],
    _,
) = precision_recall_fscore_support(
    y_true_firststage, y_pred_firststage, average="binary"
)
scores["acc"] = accuracy_score(y_true_firststage, y_pred_firststage)
scores_non_bin = {}
(
    scores_non_bin["precision"],
    scores_non_bin["recall"],
    scores_non_bin["fscore"],
    _,
) = precision_recall_fscore_support(
    y_true_firststage, y_pred_firststage, average="macro"
)
scores_non_bin2 = {}
(
    scores_non_bin2["precision"],
    scores_non_bin2["recall"],
    scores_non_bin2["fscore"],
    _,
) = precision_recall_fscore_support(
    y_true_firststage, y_pred_firststage, average="micro"
)
print(y_fp)
print("first-stage macro-avg: ", round_dict_vals(scores_non_bin))
print("first-stage micro-avg: ", round_dict_vals(scores_non_bin2))
print("first-stage binary-avg: ", round_dict_vals(scores))

# second stage true score
y_true_secondstage_idc = np.where(y_true > -1)[
    0
]  # idc of true bullying instances to be classified in second stage
y_true_secondstage = y_true[y_true_secondstage_idc]
y_pred_secondstage = y_pred[y_true_secondstage_idc]
scores_secondstage = {}
(
    scores_secondstage["precision"],
    scores_secondstage["recall"],
    scores_secondstage["fscore"],
    _,
) = precision_recall_fscore_support(
    y_true_secondstage, y_pred_secondstage, average="macro"
)
scores_secondstage["acc"] = accuracy_score(y_true_secondstage, y_pred_secondstage)

print("second-stage macro-avg: ", round_dict_vals(scores_secondstage))

# second stage no misclassified in first stage
correct_firststage_idc = np.where(np.equal(y_true_firststage, y_pred_firststage))[
    0
]  # filter erroneous first-stage predictions
y_true_correct_secondstage_intermed = y_true[correct_firststage_idc]
y_pred_correct_secondstage_intermed = y_pred[correct_firststage_idc]

y_true_correct_secondstage_idc = np.where(y_true_correct_secondstage_intermed > -1)[
    0
]  # drop non-bullying instances to asses second stage role classifier on correct in first phase
y_true_correct_secondstage = y_true_correct_secondstage_intermed[
    y_true_correct_secondstage_idc
]
y_pred_correct_secondstage = y_pred_correct_secondstage_intermed[
    y_true_correct_secondstage_idc
]

scores_secondstage_noerrors = {}
(
    scores_secondstage_noerrors["precision"],
    scores_secondstage_noerrors["recall"],
    scores_secondstage_noerrors["fscore"],
    _,
) = precision_recall_fscore_support(
    y_true_correct_secondstage, y_pred_correct_secondstage, average="macro"
)
scores_secondstage_noerrors["acc"] = accuracy_score(
    y_true_correct_secondstage, y_pred_correct_secondstage
)

scores_secondstage_noerrors2 = {}
(
    scores_secondstage_noerrors2["precision"],
    scores_secondstage_noerrors2["recall"],
    scores_secondstage_noerrors2["fscore"],
    _,
) = precision_recall_fscore_support(
    y_true_correct_secondstage, y_pred_correct_secondstage, average="micro"
)

print(
    "second-stage without erroneous pred first-stage macro-avg: ",
    round_dict_vals(scores_secondstage_noerrors),
)
print(
    "second-stage without erroneous pred first-stage micro-avg: ",
    round_dict_vals(scores_secondstage_noerrors2),
)
