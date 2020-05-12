from __future__ import division
from sklearn.externals.joblib import load
import numpy as np
import os
import json
import pprint
import util
import glob
import cPickle as pickle
from itertools import groupby, ifilter
import inspect
from terminaltables import AsciiTable
from sklearn.metrics import confusion_matrix
import random
import math
import sys
import functools
from sklearn.preprocessing import Normalizer, MaxAbsScaler
import itertools
import datahandler as dh
from sklearn.externals.joblib import load
import settings as s
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.pipeline import Pipeline as SkPipeline


def get_y_by_label(docs, hits=[]):
    ys_true = []
    ys_pred = []
    for id, [y_true, y_pred, labels] in docs.iteritems():
        if any(i in hits for i in labels):
            ys_true.append(y_true)
            ys_pred.append(y_pred)
    return ys_true, ys_pred


def summarize_cv(cv):
    cv_res = cv.cv_results_
    allowed_keys = [
        "rank_test_score",
        "params",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
    ]
    assert np.unique([len(cv_res[k]) for k in allowed_keys]).size == 1
    header = [k for k in cv_res.keys() if k in allowed_keys]
    report = "{}\n".format("\t".join(header))
    res = zip(*cv_res.values())
    res.sort(key=lambda x: x[0])
    allowed_idc = [i for i, k in enumerate(cv_res.keys()) if k in allowed_keys]

    for result in res:
        allowed_res = np.array(result)[allowed_idc]
        report += "{}\n".format(
            "\t".join(
                [str(el).replace("\n", " ").replace("\t", " ") for el in allowed_res]
            )
        )

    return report


def get_metadata(log_dir):
    all_pipe_metadata = []
    for foldscore_fp in os.listdir(log_dir):
        json_fp = os.path.join(log_dir, foldscore_fp)
        with open(json_fp, "rt") as jsonf:
            for l in jsonf:
                pipe_metadata = json.loads(l)
                all_pipe_metadata.append(pipe_metadata)

    return all_pipe_metadata


def sort_best(metadata, metric, n=10):
    metadata.sort(key=lambda x: x["results"][metric], reverse=True)
    return metadata


def calculate_scores(y_true, y_pred):
    labels = sorted(np.unique(np.append(y_true, y_pred)))
    scores = {}
    (
        scores["precision"],
        scores["recall"],
        scores["fscore"],
        _,
    ) = precision_recall_fscore_support(y_true, y_pred, average=s.SCORE_AVERAGING)
    scores["acc"] = accuracy_score(y_true, y_pred)

    (
        scores["precision_all_labels"],
        scores["recall_all_labels"],
        scores["fscore_all_labels"],
        _,
    ) = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
    if s.MULTICLASS:
        scores["auc"] = None
    else:
        scores["auc"] = roc_auc_score(y_true, y_pred)
    return scores


def bootstrap_resample(
    y_true, y_pred, n, sample_size, bootstrapping=True, replace=True
):
    import numpy as np

    remaining_indices = range(y_true.shape[0])
    res = []
    for i in range(n):
        if (i % (n // 5)) == 0:
            print "Sample %d of %d..." % (i + 1, n)
        random_sample_indices = np.random.choice(
            remaining_indices, size=sample_size, replace=replace
        )
        if not bootstrapping:
            remaining_indices = [
                i for i in remaining_indices if i not in random_sample_indices
            ]
        y_true_resamp = y_true[random_sample_indices]
        y_pred_resamp = y_pred[random_sample_indices]
        scores = calculate_scores(y_true_resamp, y_pred_resamp)
        res.append(scores)

    return res


def plot_histogram(data, fp):

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    plt.xlim([min(data), max(data)])
    num_bins = 100
    bins = np.linspace(np.min(data), np.max(data), num=num_bins)
    plt.hist(
        data.ravel(), bins="auto",
    )
    mean = data.mean()
    plt.axvline(mean, color="b", linestyle="dashed", linewidth=1)
    plt.title("Bootstrap resampled scores on heldout testset")
    plt.xlabel("F1score")
    plt.ylabel("Bootstrap sample count")

    plt.savefig(fp)
    plt.gcf().clear()
    plt.clf()
    plt.close("all")


def random_baseline(y_true):

    # generate random y's with values in y and len = y.shape[0]
    sample_values = np.unique(y_true)
    y_randbl = np.random.choice(sample_values, size=y_true.shape[0], replace=True)
    scores = calculate_scores(y_true, y_randbl)

    return scores


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.round(cm_norm, decimals=3) * 100
        print ("Normalized confusion matrix")
        print (cm_norm)

        plt.imshow(cm_norm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
            thresh = cm_norm.max() / 2
            plt.text(
                j,
                i,
                "{}%\n($n={}$)".format(cm_norm[i, j], cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    else:
        print ("Confusion matrix, without normalization")
        print (cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            thresh = cm.max() / 2
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def majority_baseline(y_true):

    unique, counts = np.unique(y_true, return_counts=True)
    maj_value = unique[counts.tolist().index(max(counts.tolist()))]
    y_maj = np.full(y_true.shape, maj_value)
    scores = calculate_scores(y_true, y_maj)

    return scores


def savefig_confusion_matrix(y_true, y_pred, fp):
    # plot confusion matrix
    cm_norm_plot_fp = fp.replace(".png", "_norm.png")

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    class_names = [s.LABELMAP[val] for val in np.unique(y_true)]
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix, classes=class_names, title="Confusion matrix, without normalization"
    )
    plt.savefig(fp)
    plt.gcf().clear()
    plt.clf()
    plt.close("all")

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        normalize=True,
        title="Normalized confusion matrix",
    )
    plt.savefig(cm_norm_plot_fp)
    plt.gcf().clear()
    plt.clf()
    plt.close("all")

    print ("Saved confusion matrix plot to {}".format(fp))


def split_lines_by_idc_from_datafile(fp, out_idc):
    def parse_svmlight_comment(inst_line):

        comment = inst_line.split("#")[-1].strip()
        labels = literal_eval("[{}".format(comment.split(" [")[-1]))
        return labels

    with open(fp, "rt") as f:
        lines = f.readlines()

    labels = [parse_svmlight_comment(lines[i]) for i in out_idc]
    return labels


def get_y_per_type(labels, instances):

    inst_cnt = len(instances)

    y_per_type = dict.fromkeys(
        labels.keys(), {"y_true": [], "y_pred": []}
    )  # {"insult": ("y_true": y_true, "y_pred": y_pred)}
    y_per_type["no_type"] = {"y_true": [], "y_pred": []}

    # get all label types: for each instance check which labels match, or notype, add ys to relevant place in y_per_type
    for label_type, label_hits in labels.iteritems():

        if (
            label_type != "no_type"
        ):  # an artifact from dict.fromkeys to overwrite the init object
            y_per_type[label_type] = {"y_true": [], "y_pred": []}

        for i, (y_true, y_pred, inst_labels) in enumerate(instances):

            if any(i in inst_labels for i in label_hits):

                y_per_type[label_type]["y_true"].append(y_true)
                y_per_type[label_type]["y_pred"].append(y_pred)

            elif inst_labels == []:
                y_per_type["no_type"]["y_true"].append(y_true)
                y_per_type["no_type"]["y_pred"].append(y_pred)
                del instances[i]

    # compute scores
    more_than_one_label = inst_cnt - sum(
        len(v["y_true"]) for v in y_per_type.itervalues()
    )

    error_rates = {}
    print (
        "label:\tn instances of label\terror rate.\n----------------------------------------"
    )
    for label, y_dict in y_per_type.iteritems():
        error_rate = 1 - accuracy_score(y_dict["y_true"], y_dict["y_pred"])
        error_rates[label] = error_rate
        print ("{}:\t{}\t{}".format(label, len(y_dict["y_true"]), error_rate))
    pass


def make_error_rate_per_label_report(y_true, y_pred, fp, out_idc, labels):

    # make a list of instance tuples [(y_true, y_pred, labels), ...]
    labels_out = split_lines_by_idc_from_datafile(fp, out_idc)
    instances = zip(y_true, y_pred, labels_out)

    # make error rate matrix
    y_per_type = get_y_per_type(labels, instances)


def filter_folds_by_name(folds, name, best_params):

    filt_folds = []
    for fold in folds:
        pipe_name = fold["pipe_name"]
        reconstr_name = "".join(
            [c for c in pipe_name if c.isupper() or not c.isalnum()]
        ).lower()
        # # fix for wrong naming in early runs
        # if pipe_name.count('linearsvc_') >= 2:
        #     reconstr_name = pipe_name.split('linearsvc_')[0]+'linearsvc'
        # print(pipe_name, reconstr_name, name)
        if reconstr_name == name:
            try:
                model = load(fold["savepath_model"])
                model_params = model.get_params()
                if (
                    "cascad" in name
                    or best_params.viewitems() <= model_params.viewitems()
                ):
                    filt_folds.append(fold)

            except Exception as e:
                print (
                    "Could not load model {}. {}".format(
                        fold["savepath_model"], e.message
                    )
                )

    return filt_folds


def tuple_counts_to_percents(inputList):
    total = sum(x[1] for x in inputList)
    return [(x[0], round((100.0 * x[1] / total), ndigits=2)) for x in inputList]


def print_result_table(
    all_runs,
    fold_meta,
    all_bootstrap_score,
    score_rand_bl_avg,
    score_maj_bl_avg,
    METRIC,
):
    # compare the effect of the pipeline steps by classifier type
    tabledata = [
        [
            "clfalgo",
            "clf f1",
            "p",
            "r",
            "acc",
            "+fs f1",
            "p",
            "r",
            "acc",
            "+res f1",
            "p",
            "r",
            "acc",
            "fs+res f1",
            "p",
            "r",
            "acc",
            "res+fs f1",
            "p",
            "r",
            "acc",
        ]
    ]

    for k, g in groupby(
        sorted(all_runs, key=lambda x: x.pipe_name_.split("+")[-1]),
        key=lambda x: x.pipe_name_.split("+")[-1],
    ):
        row = [k, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # run_folds = [value for key, value in all_fold_meta.iteritems() if k.replace("classifier", "") in key.replace(
        #     "DECTREE", "decisiontree").replace(
        #     "LOGRES", "logisticregression").replace(
        #     "PASSAGGR", "passiveaggressive").lower()][0] # get the folds that match the run

        run_folds = [
            fold
            for fold in ifilter(
                lambda x: k.lower() in x["pipe_name"].lower(), fold_meta
            )
        ]  # filter allfold meta by classifier in pipe name

        for cv in g:
            folds_by_name = filter_folds_by_name(
                run_folds, cv.pipe_name_, cv.best_params_
            )
            # match all all_fold_meta folds entry with the pipename and collect their score dicts
            fold_pipe_scores = [
                calculate_scores(fold["y_true"], fold["y_pred"])
                for fold in folds_by_name
            ]
            # fold_pipe_scores = [fold["results"] for fold in filter_folds_by_name(run_folds, cv.pipe_name_, cv.best_params_)]
            cv_scores = {}
            score_keys = fold_pipe_scores[0].keys()
            for k in score_keys:
                cv_scores[k] = [score[k] for score in fold_pipe_scores]
                if "_all_labels" in k:
                    cv_scores["{}_avg".format(k)] = np.nanmean(
                        np.array(cv_scores[k], dtype=np.float), axis=0
                    )
                else:
                    cv_scores["{}_avg".format(k)] = np.nanmean(
                        np.array(cv_scores[k], dtype=np.float)
                    )

            print (
                "\n============={} best CV score + best params + CV scores by label=============".format(
                    cv.pipe_name_.upper()
                )
            )
            print (cv.pipe_name_, cv.best_score_)
            print (cv.best_params_)
            print ("\nMETRIC\tCV\tHOLDOUT")
            for k, v in cv_scores.iteritems():
                if "_all_labels_avg" in k:
                    print (
                        "{}\t{}\t{}".format(k, v, all_bootstrap_score[cv.pipe_name_][k])
                    )

            # fill the  table
            steps = "+".join([step[0] for step in cv.estimator.steps])
            if re.search(r"^(classify\d*\+?)+", steps):
                i = 1
            elif re.search(r"^(featselect\d*\+?)+(classify\d*\+?)+", steps):
                i = 5
            elif re.search(r"^(resample\d*\+?)+(classify\d*\+?)+", steps):
                i = 9
            elif re.search(
                r"^(featselect\d*\+?)+(resample\d*\+?)+(classify\d*\+?)+", steps
            ):
                i = 13
            elif re.search(
                r"^(resample\d*\+?)+(featselect\d*\+?)+(classify\d*\+?)+", steps
            ):
                i = 17

            cv_score = np.round(cv.best_score_, decimals=4) * 100
            if (
                row[i] < cv_score
            ):  # this to make sure the best/highest score for each fs parametrisation is choosen
                row[i] = np.round(cv_scores["fscore_avg"], decimals=4) * 100
                row[i + 1] = np.round(cv_scores["precision_avg"], decimals=4) * 100
                row[i + 2] = np.round(cv_scores["recall_avg"], decimals=4) * 100
                row[i + 3] = np.round(cv_scores["acc_avg"], decimals=4) * 100

                if BOOTSTRAP_RESAMP:
                    row[i] = "{} | {}".format(
                        row[i],
                        np.round(
                            all_bootstrap_score[cv.pipe_name_][
                                "fscore_avg".format(METRIC)
                            ],
                            decimals=4,
                        )
                        * 100,
                    )
                    row[i + 1] = "{} | {}".format(
                        row[i + 1],
                        np.round(
                            all_bootstrap_score[cv.pipe_name_][
                                "precision_avg".format(METRIC)
                            ],
                            decimals=4,
                        )
                        * 100,
                    )
                    row[i + 2] = "{} | {}".format(
                        row[i + 2],
                        np.round(
                            all_bootstrap_score[cv.pipe_name_][
                                "recall_avg".format(METRIC)
                            ],
                            decimals=4,
                        )
                        * 100,
                    )
                    row[i + 3] = "{} | {}".format(
                        row[i + 3],
                        np.round(
                            all_bootstrap_score[cv.pipe_name_][
                                "acc_avg".format(METRIC)
                            ],
                            decimals=4,
                        )
                        * 10,
                    )
        tabledata.append(row)

    # add the baselines
    tabledata.append(
        [
            "random bl",
            np.round(score_rand_bl_avg["fscore"], decimals=4) * 100,
            np.round(score_rand_bl_avg["precision"], decimals=4) * 100,
            np.round(score_rand_bl_avg["recall"], decimals=4) * 100,
            np.round(score_rand_bl_avg["acc"], decimals=4) * 100,
        ]
    )
    tabledata.append(
        [
            "majority bl",
            np.round(score_maj_bl_avg["fscore"], decimals=4) * 100,
            np.round(score_maj_bl_avg["precision"], decimals=4) * 100,
            np.round(score_maj_bl_avg["recall"], decimals=4) * 100,
            np.round(score_maj_bl_avg["acc"], decimals=4) * 100,
        ]
    )

    table = AsciiTable(tabledata, title="Crossvalidation macro-avg score")
    print ("\n\n")
    print (table.table)


if __name__ == "__main__":

    BOOTSTRAP_RESAMP = False
    CONFUSION_MATRIX = False
    LABEL_ERROR_RATE = True

    N_SAMPLES = 10000
    # N_SAMPLES = 10

    LANGUAGE = "en"
    METRIC = ["fscore", "precision", "recall", "acc"]

    run_dirs = {
        "en_detect": "/home/gilles/repos/cbrole/static/DETECT_LSVC_en_detect_3_CBROLE_150428295005",
        "nl_detect": "/home/gilles/repos/cbrole/static/LINEARSVC_3_CBEVENT149968157182_nl",
        "en_old": [
            # '/home/gilles/repos/cbrole/static/LINEARSVC_3_CBROLE149381838742_cbrole_en',
            "/home/gilles/repos/cbrole/static/LINEARSVC_3_CBROLE150106657194_cbrole_en",  # fixed run
            "/home/gilles/repos/cbrole/static/PASSAGGR_3_CBROLE149545164481_cbrole_en",
            "/home/gilles/repos/cbrole/static/DECISIONTREE_3_CBROLE149639710957_cbrole_en",
            "/home/gilles/repos/cbrole/static/SGD_3_CBROLE149509869273_cbrole_en",
            "/home/gilles/repos/cbrole/static/LOGRES_3_CBROLE14945828729_cbrole_en",
            "/home/gilles/repos/cbrole/static/VOTING_3_CBROLE149942077331_cbrole_en",
        ],
        "nl_old": [
            "/home/gilles/repos/cbrole/static/LINEARSVC_3_CBROLE149744106886_cbrole_nl",
            "/home/gilles/repos/cbrole/static/PASSAGGR_3_CBROLE149821892828_cbrole_nl",
            "/home/gilles/repos/cbrole/static/DECTREE_3_CBROLE149872537341_cbrole_nl",
            "/home/gilles/repos/cbrole/static/SGD_3_CBROLE14992635396_cbrole_nl",
            "/home/gilles/repos/cbrole/static/LOGRES_3_CBROLE14978597435_cbrole_nl",
            "/home/gilles/repos/cbrole/static/VOTING_3_CBROLE149960085407_cbrole_nl",
        ],
        "en": "/home/gilles/repos/cbrole/static/BASIC_en_3_CBROLE_150236394374",
        # "/home/gilles/repos/cbrole/static/VOTING_en_3_CBROLE_150426243867"
        # "/home/gilles/repos/cbrole/static/en_CASCADE_3CBROLE_150894970996",
        "nl":
        # "/home/gilles/repos/cbrole/static/NL_SGD_nl_3_CBROLE_150868441264",
        # "/home/gilles/repos/cbrole/static/NL_LOGRESPASSAGG_nl_3_CBROLE_150749638151",
        # "/home/gilles/repos/cbrole/static/NL_LINEARSVC_nl_3_CBROLE_150705402137",
        # "/home/gilles/repos/cbrole/static/nl_VOTING_3CBROLE_15099616772",
        "/home/gilles/repos/cbrole/static/nl_VOTING_3CBROLE_150999028765",  # THIS IS THE CASCADING CLF
        # "/home/gilles/repos/cbrole/static/REST_nl_3_CBROLE_150686533984",
    }

    type_labels = {
        "threat": ["Threat_Blackmail"],
        "insult": ["General_insult", "Attacking_relatives", "Sexism", "Racism"],
        "curse": ["Curse_Exclusion"],
        "defamation": ["Defamation"],
        "sexual": ["Harmless_sexual_talk", "Sexual_harassment"],
        "defense": [
            "Good_characteristics",
            "General_defense",
            "Assertive_selfdef",
            "Powerless_selfdef",
        ],
        "encouragement": ["Encouraging_harasser"],
    }

    # all_labels = {'all': functools.partial(get_all_y)}  # for unit testing gets the whole holdout
    #
    # event_labels = {'event': functools.partial(get_y_by_label, hits=["1_Victim", "1_Bystander_defender", "1_Harasser",
    #                                                                  "1_Bystander_assistant", "2_Victim",
    #                                                                  "2_Bystander_defender", "2_Harasser",
    #                                                                  "2_Bystander_assistant"])}
    LABELS = type_labels

    DATA_FP = s.langspec[LANGUAGE]["DATA_FP"]  # No longer needed
    run_dir = run_dirs[LANGUAGE]

    all_runs = []
    all_bl = {
        "rand_baseline": [],
        "maj_baseline": [],
    }
    all_bootstrap_score = {}
    y_distr = {}

    X_in, y_in = dh.load_data("{}/holdin.svm".format(run_dir), memmapped=False)
    X_out, y_true = dh.load_data("{}/holdout.svm".format(run_dir), memmapped=False)

    fold_log_dirp = glob.glob("{}/fold_log".format(run_dir))[0]
    fold_meta = get_metadata(fold_log_dirp)

    split = json.load(open("{}/holdinout_split_indices.json".format(run_dir), "rt"))
    out_idc = split["holdout"]
    in_idc = split["holdin"]

    full_class_counts = np.asarray(
        np.unique(np.append(y_in, y_true), return_counts=True)
    ).T.tolist()
    in_class_counts = np.asarray(np.unique(y_in, return_counts=True)).T.tolist()

    out_class_counts = np.asarray(np.unique(y_true, return_counts=True)).T.tolist()
    print (
        "{} HOLDIN Class distribution: {} (n={}).".format(
            run_dir, tuple_counts_to_percents(in_class_counts), in_class_counts
        )
    )
    print (
        "{} HOLDOUTClass distribution: {} (n={}).".format(
            run_dir, tuple_counts_to_percents(out_class_counts), out_class_counts
        )
    )
    y_distr[run_dir] = {
        "fullset": (tuple_counts_to_percents(full_class_counts), full_class_counts),
        "holdout": (tuple_counts_to_percents(out_class_counts), out_class_counts),
        "holdin": (tuple_counts_to_percents(in_class_counts), in_class_counts),
    }

    rand_bl_score = random_baseline(y_true)  # Do I have to resample these too
    maj_bl_score = majority_baseline(y_true)

    all_bl["rand_baseline"].append((rand_bl_score, run_dir))
    all_bl["maj_baseline"].append((maj_bl_score, run_dir))

    y_fps = glob.glob("{}/cv_pipelines/*/y_out_true-y_out_pred.json".format(run_dir))

    for y_fp in y_fps:

        y_out_pred_y_out_true = json.load(open(y_fp, "rt"))
        y_pred = np.array(y_out_pred_y_out_true["y_out_pred"])
        y_out_true = np.array(y_out_pred_y_out_true["y_out_true"])
        assert np.array_equal(y_true, y_out_true)
        pipe_name = os.path.basename(os.path.abspath(os.path.join(y_fp, os.pardir)))

        # Error rate per label report
        if LABEL_ERROR_RATE:
            print (
                "\n====+====+====+ {} ERROR RATE PER SUBTYPE +====+====+====".format(
                    pipe_name.upper()
                )
            )
            make_error_rate_per_label_report(y_true, y_pred, DATA_FP, out_idc, LABELS)

        # Confusion matrix
        if CONFUSION_MATRIX:
            fn = os.path.basename(y_fp).replace(
                "y_out_true-y_out_pred.json", "confusionmatrix.png"
            )
            cm_plot_fp = os.path.join(
                os.path.abspath(os.path.join(y_fp, os.pardir)), fn
            )
            savefig_confusion_matrix(y_true, y_pred, cm_plot_fp)

        # Bootstrap resampling
        if BOOTSTRAP_RESAMP:
            print (
                "\n====+====+====+ {} BOOTSTRAP RESAMPLING +====+====+====".format(
                    pipe_name.upper()
                )
            )
            sample_size = y_true.shape[0]
            n = N_SAMPLES
            bootstrapresample_scores = bootstrap_resample(
                y_true, y_pred, n, sample_size
            )

            score_keys = bootstrapresample_scores[0].keys()
            bootstrap_score = {}
            for k in score_keys:
                bootstrap_score[k] = [score[k] for score in bootstrapresample_scores]
                if "_all_labels" in k:
                    bootstrap_score["{}_avg".format(k)] = np.nanmean(
                        np.array(bootstrap_score[k], dtype=np.float), axis=0
                    )
                else:
                    bootstrap_score["{}_avg".format(k)] = np.nanmean(
                        np.array(bootstrap_score[k], dtype=np.float)
                    )

            all_bootstrap_score[pipe_name] = bootstrap_score

            for metric in METRIC:
                fn = os.path.basename(y_fp).replace(
                    "y_out_true-y_out_pred.json", "{}_hist.png".format(metric.upper())
                )
                plot_fp = os.path.join(
                    os.path.abspath(os.path.join(y_fp, os.pardir)), fn
                )
                plot_histogram(bootstrap_score[metric], plot_fp)

    # Show the ten best parametrised pipelines and their scores. This is the basic functionality we want here.
    cv_pipelines_parametrised = []
    cv_pipeline_fp = glob.glob("{}/cv_pipelines/*/*.joblibpkl".format(run_dir))
    pipe_res = {}
    for cvpipefp in cv_pipeline_fp:
        pipe_name = os.path.basename(os.path.abspath(os.path.join(cvpipefp, os.pardir)))
        cv_pipeline = load(open(cvpipefp, "rb"))
        cv_pipeline.pipe_name_ = pipe_name
        cv_pipelines_parametrised.append(cv_pipeline)
    cv_pipelines_parametrised.sort(key=lambda x: x.best_score_, reverse=True)

    all_runs.extend(cv_pipelines_parametrised)

    n = 5
    for i, cvpp in enumerate(cv_pipelines_parametrised[:n]):
        if i == 0:
            print (
                "\n+++++================+++++ {} TOP {} BEST PIPELINE PARAMS +++++================+++++".format(
                    cvpp.pipe_name_.split("+")[-1], n
                )
            )
        print (
            "{}) {}: {}\n{}\n{}".format(
                i,
                cvpp.pipe_name_,
                cvpp.best_score_,
                cvpp.best_estimator_.steps,
                cvpp.best_params_,
            )
        )

    # # PRINT CLASS DISTRIBUTIONS FOR VERIFICATION
    #     print("==========HOLDOUT CLASS DISTRIBUTIONS=========")
    #     for k, v in y_distr.iteritems():
    #         print("{}\t{}".format(k, v))

    # make avg baseline score
    score_rand_bl_avg = {}
    score_maj_bl_avg = {}

    for metric in METRIC:
        score_rand_bl_avg[metric] = sum(
            score[0][metric] for score in all_bl["rand_baseline"]
        ) / len(all_bl["rand_baseline"])
        score_maj_bl_avg[metric] = sum(
            score[0][metric] for score in all_bl["maj_baseline"]
        ) / len(all_bl["maj_baseline"])

    all_runs.sort(key=lambda x: x.best_score_, reverse=True)
    print (
        "\n+++++=================+++++================+++++=================+++++================+++++=================+++++================+++++"
    )
    for i, cvpp in enumerate(all_runs[:10]):
        print (
            "{}) ".format(i),
            cvpp.pipe_name_,
            cvpp.best_score_,
            cvpp.best_estimator_.steps,
            cvpp.best_params_,
        )

    # WE WANT TO KNOW HOW MANY FOLDS HAVE FAILED AKA SET TO 0 for each system
    pipe_check = {}  # collect run name for each pipe
    for clfname, g in groupby(
        sorted(all_runs, key=lambda x: x.pipe_name_.split("+")[-1]),
        key=lambda x: x.pipe_name_.split("+")[-1],
    ):
        pipe_check[clfname] = []
        print ("\n============={}=============".format(clfname))
        for cv in g:
            pipe_check[clfname].append(cv.pipe_name_)
            split_test_scores = [v for k, v in cv.cv_results_.items() if "split" in k]
            split_test_scores_zerocount = [
                spl.tolist().count(0.0) for spl in split_test_scores
            ]
            if any(s > 0 for s in split_test_scores_zerocount):
                print (
                    "{} has {} folds with zero score. {} parametrisations total for which the zero counts for nth fold are {}. These parametrisations have produced an error likely and will be ignored for determining the CV winner.".format(
                        cv.pipe_name_,
                        sum(i > 0 for i in split_test_scores_zerocount),
                        len(split_test_scores[0]),
                        split_test_scores_zerocount,
                    )
                )
            else:
                print (
                    "{} has {} folds with zero score.".format(
                        cv.pipe_name_, sum(i > 0 for i in split_test_scores_zerocount)
                    )
                )

    # write the partial run info pipe names for reuse in fix
    partialinfo = util.flatten(pipe_check.values())
    json.dump(partialinfo, open(os.path.join(run_dir, "partialruninfo.json"), "wt"))

    print_result_table(
        all_runs,
        fold_meta,
        all_bootstrap_score,
        score_rand_bl_avg,
        score_maj_bl_avg,
        METRIC,
    )

    # # Binomial test works
    # def binomial_test(s, n, p=0.5):
    #     import decimal
    #     decimal.getcontext().prec = 10
    #
    #     def f(s, n, p):
    #         s = decimal.Decimal(s)
    #         n = decimal.Decimal(n)
    #         p = decimal.Decimal(p)
    #         return (decimal.Decimal(math.factorial(n))/decimal.Decimal(math.factorial(s)*math.factorial(n-s)))*(p**n)
    #
    #     def sigma(func, frm, to, p):
    #         result = decimal.Decimal(0)
    #         for i in range(frm, to + 1):
    #             result += func(i, to, p)
    #         return result
    #
    #     return sigma(f, s, n, p)
    #
    # print(sys.float_info)
    # print(binomial_test(520, 1000))
    # print(3/2)
