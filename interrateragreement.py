#!/usr/bin/env python3
"""
interrateragreement.py
cbrole 
12/11/17
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from itertools import groupby

pd.set_option("display.max_colwidth", -1)

from nltk.metrics.agreement import AnnotationTask
import nltk


def filter_label(lab):
    """
    Removes the severity indication "1_/2_" from the role labels.
    :param lab: label as string.
    :return: filtered label
    """
    if "_" in lab:
        lab = lab.split("_", 1)[1]
    return lab


def parse_data(fp):
    data = []
    annotators = {}
    with open(fp, "rt") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i != 0:
                linesplit = line.split("\t")
                postid = linesplit[0]
                for i, annlab in enumerate(linesplit[1:]):
                    annlab = filter_label(annlab)
                    labinst = (annotators[i], postid, annlab)
                    data.append(labinst)
            else:
                headersplit = line.split("\t")
                for i, annttr in enumerate(headersplit[1:]):
                    annotators[i] = annttr
    return data


def write_multi_dfs(df_list, file_name, sheets="sheet1", spaces=1):
    writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheets, startrow=row, startcol=0)
        row = row + len(dataframe.index) + spaces + 1
    writer.save()


class CustomAnnotationTask(AnnotationTask):
    def __init__(self, *args, **kwargs):
        super(CustomAnnotationTask, self).__init__(*args, **kwargs)
        self.label_encoder = LabelEncoder().fit(np.array(list(self.K)))
        self.sk_labels = self._get_scikit_labels()

    def _get_scikit_labels(self):
        sk_labels = []
        key = lambda x: x["coder"]
        data = self.data[:]
        data.sort(key=key)
        for item, item_data in groupby(data, key=key):
            labels_ann = [idat["labels"] for idat in item_data]
            sk_labels.append(labels_ann)
        return sk_labels

    def scikit_metric_pairwise(self, func, **kwargs):
        total = []
        s = self.sk_labels[:]
        for lab1 in self.sk_labels:
            s.remove(lab1)
            for lab2 in s:
                total.append(getattr(sklearn.metrics, func)(lab1, lab2, **kwargs))
        ret = np.mean(total, axis=0)
        return ret


if __name__ == "__main__":
    libstatement = "{} Computed with nltk.metrics.agreement.AnnotatorTask from nltk version {}.".format(
        datetime.now().strftime("%d %b %Y"), nltk.__version__
    )
    data_fp = [
        "/home/gilles/repos/cbrole/cbrole/finalresults+IAA/IAA_JASIST17/IAA_JASIST17/IAA_scores_roles/NL/CB_NL_roles_metadata.txt",
        "/home/gilles/repos/cbrole/cbrole/finalresults+IAA/IAA_JASIST17/IAA_JASIST17/IAA_scores_roles/EN/CB_EN_roles_metadata.txt",
    ]
    dataset = [
        (os.path.splitext(os.path.basename(fp))[0], parse_data(fp)) for fp in data_fp
    ]
    pass
    # dataset = [("test", [
    #     ("Annotator1", "id1", 0),
    #     ("Annotator2", "id1", 0),
    #     ("Annotator3", "id1", 0),
    #
    #     ("Annotator1", "id2", 1),
    #     ("Annotator2", "id2", 1),
    #     ("Annotator3", "id2", 1),
    #
    #     ("Annotator1", "id3", 2),
    #     ("Annotator2", "id3", 2),
    #     ("Annotator3", "id3", 2),
    # ])]
    metrics = [
        "f1_score",
        "recall_score",
        "precision_score",
        "kappa",
        "multi_kappa",
        "alpha",
        "pi",
        "S",
    ]
    sklearn_kwargs = {
        "f1_score": [
            {"average": None},
            {"average": "macro"},
            {"average": "micro"},
            {"average": "weighted"},
            # {"average": "samples"},
        ],
        "recall_score": [
            {"average": None},
            {"average": "macro"},
            {"average": "micro"},
            {"average": "weighted"},
            # {"average": "samples"},
        ],
        "precision_score": [
            {"average": None},
            {"average": "macro"},
            {"average": "micro"},
            {"average": "weighted"},
            # {"average": "samples"},
        ],
    }

    replace_lab = {"Bystander_assistant": "Harasser"}
    dataset_with_exp = []
    for dataset_name, data in dataset:  # for replacing labels or collapsing labels
        dataset_with_exp.append((dataset_name, data))
        data_replace = [
            (d[0], d[1], replace_lab[d[2]])
            if d[2] in replace_lab
            else (d[0], d[1], d[2])
            for d in data
        ]
        dataset_with_exp.append((dataset_name + "_replace", data_replace))

    # check labels:
    print(np.unique([d[2] for d in dataset[1][1]]))
    print(np.unique([d[2] for d in dataset_with_exp[2][1]]))

    # df = pd.DataFrame(columns=metrics)
    all_df = []

    for dataset_name, data in dataset_with_exp:
        result = OrderedDict()
        result["dataset"] = dataset_name
        anntask = CustomAnnotationTask(data=data)
        df = pd.DataFrame(columns=metrics)
        for func in metrics:
            if func in [
                method_name
                for method_name in dir(anntask)
                if callable(getattr(anntask, method_name))
            ]:  # func is built-in of AnnotationTask class
                res = getattr(anntask, func)()
                funcname = func
                if funcname in result:
                    result[funcname].append(res)
                else:
                    result[funcname] = [res]

            else:  # else we use one of sklearn metrics
                for kwargs_orig in sklearn_kwargs[func]:
                    kwargs = kwargs_orig.copy()
                    if "labels" not in kwargs:
                        kwargs["labels"] = list(anntask.K)
                    funcname = "{}_{}".format(
                        func,
                        "_".join(
                            ["{}={}".format(k, str(v)) for k, v in kwargs.items()]
                        ),
                    )
                    try:
                        res = anntask.scikit_metric_pairwise(func, **kwargs)
                    except Exception as e:
                        print(e)
                        res = "Failed {} with {}.".format(func, kwargs)
                        pass
                    if funcname in result:
                        result[funcname].append(res)
                    else:
                        result[funcname] = [res]

        df = df.from_dict(result)
        df = df.set_index("dataset").transpose()
        all_df.append(df)
        print(df.to_string())
    all_df.append(df.from_dict({"libstatement": [libstatement]}).transpose())
    write_multi_dfs(all_df, "iaascores.xlsx")
