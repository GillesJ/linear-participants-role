#!/usr/bin/env python3
"""
Script for generating LaTeX pfg code for the paper.

generate_confusion_matrix.py
cbrole 
2/26/18
Copyright (c) Gilles Jacobs. All rights reserved.
"""
import numpy as np
import matplotlib as mpl
import json
import settings as s
import itertools
import os
from sklearn.metrics import confusion_matrix

mpl.use("pgf")

# I make my own newfig and savefig functions
def figsize(scale):
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,  # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ],
}

mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    figure_scale=1.0,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    plt.figure(figsize=figsize(figure_scale))
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.round(cm_norm, decimals=3) * 100
        print("Normalized confusion matrix")
        print(cm_norm)

        plt.imshow(cm_norm, interpolation="nearest", cmap=cmap)
        if title:
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
        print("Confusion matrix, without normalization")
        print(cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        if title:
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

    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def savefig_confusion_matrix(fp):

    # Plot normalized confusion matrix
    plt.savefig("{}.pgf".format(fp), bbox_inches="tight")
    plt.savefig("{}.pdf".format(fp), bbox_inches="tight")
    plt.savefig("{}.svg".format(fp), bbox_inches="tight")
    print("Saved confusion matrix plot to {}".format(fp))


def plot_multiple_cm(
    cms, classes, normalize=True, title="Confusion matrix", cmap=plt.cm.Blues, width=1.0
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    fig = plt.figure(figsize=figsize(width))

    for pos, cm in enumerate(cms):
        ijk = 210 + (
            pos + 1
        )  # three separate integers describing the position of the subplot. If the three integers are I, J, and K, the subplot is the Ith plot on a grid with J rows and K columns.
        print(ijk)
        ax = fig.add_subplot(ijk)
        ax.clear()

        if normalize:
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.round(cm_norm, decimals=3) * 100
            print("Normalized confusion matrix")
            print(cm_norm)

            ax.imshow(cm_norm, interpolation="nearest", cmap=cmap)
            if title:
                ax.title(title)
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes, rotation=45)
            ax.set_yticks(tick_marks, classes)
            ax.set_yticklabels(classes, rotation=45)

            for i, j in itertools.product(
                range(cm_norm.shape[0]), range(cm_norm.shape[1])
            ):
                thresh = cm_norm.max() / 2
                ax.text(
                    j,
                    i,
                    "{}%\n($n={}$)".format(cm_norm[i, j], cm[i, j]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                )

        else:
            print("Confusion matrix, without normalization")
            print(cm)

            ax.imshow(cm, interpolation="nearest", cmap=cmap)
            if title:
                ax.title(title)
            tick_marks = np.arange(len(classes))
            ax.xticks(tick_marks, classes, rotation=45)
            ax.yticks(tick_marks, classes)

            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                thresh = cm.max() / 2
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        # plt.tight_layout()
        # plt.colorbar(cmap)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")


if __name__ == "__main__":
    # settings
    np.set_printoptions(precision=2)
    figure_scale = 0.8
    opt_dir = "/home/gilles/repos/cbrole/finalresults+IAA/figures"
    opt_fn = "confusion_matrix_paper"  # output filename suffix
    run_dirpath = {
        "EN_1_SGD+FS+RS": "/home/gilles/repos/cbrole/static/BASIC_en_3_CBROLE_150236394374/cv_pipelines/f_classif+randomundersampler+maxabsscaler+sgdclassifier/",  # 1 EN RANK
        "EN_2_CASCADE": "/home/gilles/repos/cbrole/static/en_CASCADE_3CBROLE_150894970996/cv_pipelines/maxabsscaler+cascadingclassifier/",  # 2 EN rank on holdout this is the EN cascading CLF
        "EN_3_LR+FS": "/home/gilles/repos/cbrole/static/BASIC_en_3_CBROLE_150236394374/cv_pipelines/f_classif+maxabsscaler+logisticregression/",
        "NL_1_CASCADE+FS": "/home/gilles/repos/cbrole/static/nl_VOTING_3CBROLE_150999028765/cv_pipelines/f_classif+maxabsscaler+cascadingclassifier/",  # 1 NL rank this is the NL cascading CLF
        "NL_2_LR": "/home/gilles/repos/cbrole/static/NL_LOGRESPASSAGG_nl_3_CBROLE_150749638151/cv_pipelines/maxabsscaler+logisticregression/",
        "NL_3_VOTING": "/home/gilles/repos/cbrole/static/nl_VOTING_3CBROLE_15099616772/cv_pipelines/maxabsscaler+votingclassifier/",
    }

    all_cm = []
    for pipename, run_dp in run_dirpath.items():
        # load y_true and y_pred
        with open("{}/y_out_true-y_out_pred.json".format(run_dp), "rt") as y_in:
            d = json.load(y_in)
        y_true = d["y_out_true"]
        y_pred = d["y_out_pred"]

        # generate confusion matrix data
        cnf_matrix = confusion_matrix(y_true, y_pred)
        all_cm.append(cnf_matrix)
        class_names = [s.LABELMAP[val] for val in np.unique(y_true)]
        class_names = [
            i.title() if i != "no_bullying" else "Not bullying" for i in class_names
        ]

        # plot figure
        # fig, ax = newfig(figure_scale)
        title = ""
        if pipename == "EN_1_SGD+FS+RS":
            title = "English best system (SGD + FS + RS)"
        elif pipename == "NL_1_CASCADE+FS":
            title = "Dutch best system (Cascade + FS)"

        plot_confusion_matrix(
            cnf_matrix,
            class_names,
            normalize=True,
            title=title,
            figure_scale=figure_scale,
        )
        savefig_confusion_matrix(os.path.join(run_dp, opt_fn))
        savefig_confusion_matrix(
            os.path.join(opt_dir, "{}_{}".format(pipename, opt_fn))
        )

    # plot_multiple_cm(all_cm, class_names, normalize=True,
    #                           title='')
    # savefig_confusion_matrix(os.path.join(run_dp, "{}_inone".format(opt_fn)))
