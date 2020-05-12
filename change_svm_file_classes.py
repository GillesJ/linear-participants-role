#!/usr/bin/env python
import os
from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory
import numpy as np


def make_dataset_report(fp, n_features):
    def load_data(fp, n_features=None, memmapped=False):
        """Loads a file in given vector format to CSR as required by scikit-featureselection
        """
        mem = Memory(
            ".mycache_%s" % os.path.basename(os.path.normpath(fp)), verbose=False
        )

        @mem.cache
        def get_data(in_filename, n_features):
            data = load_svmlight_file(
                in_filename, n_features=n_features, dtype=np.float32
            )
            return data[0], data[1]

        print ("Loading data...")
        X, y = get_data(fp, n_features)

        y = y.astype(int)

        print ("Data: %d instances, %d features." % (X.shape[0], X.shape[1]))
        class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
        print ("Class distribution: %s." % class_counts)

        return X, y

    X, y = load_data(fp, n_features=n_features)
    report = (
        "Dataset {} properties:\n"
        "# instances: {} x # features: {}\n"
        "Class distribution: {}.".format(
            fp,
            X.shape[0],
            X.shape[1],
            np.asarray(np.unique(y, return_counts=True)).T.tolist(),
        )
    )
    return report


def parse_line(line):
    inst_split = line.split("#")
    vector = inst_split[0].split()
    info = inst_split[1].split()
    orig_num_label = vector[0]
    orig_anns = " ".join(info[1:])
    docid = info[0]
    features = " ".join(vector[1:])

    return orig_num_label, features, docid, orig_anns


def convert_labels(orig_fp, new_fp, labels):
    if os.path.exists(new_fp):
        os.remove(new_fp)
        print "Removed existing new file."

    with open(orig_fp, "rt") as f_in, open(new_fp, "at") as f_out:

        for line in f_in:
            orig_num_label, features, docid, orig_anns = parse_line(line)
            # print(orig_num_label, docid, orig_anns)
            num_label_new = "-1"
            for (num_label, annotation) in labels:
                if any(i in orig_anns for i in annotation):
                    num_label_new = num_label
                    # print docid + ' ' + num_label + ' label found ' + orig_anns

            new_instance = "{} {} # {} {}".format(
                num_label_new, features, docid, orig_anns
            )

            # TEST
            labeldiff = len(orig_num_label) - len(num_label_new)
            old_instance = "{} {} # {} {}".format(
                orig_num_label, features, docid, orig_anns
            )
            if (
                len(old_instance) - labeldiff
                != len(line.rstrip()) - labeldiff
                != len(new_instance)
            ):
                print "WARNING: # orig chars: {} | # new instance chars: {} | old label: {} new label: {}".format(
                    len(old_instance) - labeldiff,
                    len(line.rstrip()) - labeldiff,
                    orig_num_label,
                    num_label_new,
                )

            f_out.write(new_instance + "\n")


def main():
    orig_fp = "/home/gilles/data/cbvectordata/nl_cb3/cb3_nl_bully_event.svm"
    new_fp = "/home/gilles/data/cbvectordata/nl_cb3/cb3_nl_4_roles.svm"

    # "1_Victim", "1_Bystander_defender", "1_Harasser", "1_Bystander_assistant", "2_Victim", "2_Bystander_defender", "2_Harasser", "2_Bystander_assistant"]

    threerolelabels = [
        (
            "0",
            [
                "1_Harasser",
                "2_Harasser",
                "1_Bystander_assistant",
                "2_Bystander_assistant",
            ],
        ),
        ("1", ["1_Victim", "2_Victim"]),
        ("2", ["1_Bystander_defender", "2_Bystander_defender"]),
    ]

    fourrolelabels = [
        ("0", ["1_Harasser", "2_Harasser"]),
        ("1", ["1_Victim", "2_Victim"]),
        ("2", ["1_Bystander_defender", "2_Bystander_defender"]),
        ("3", ["1_Bystander_assistant", "2_Bystander_assistant"]),
    ]

    labels = fourrolelabels
    convert_labels(orig_fp, new_fp, labels)
    new_report = make_dataset_report(new_fp, n_features=795072)
    with open(new_fp.split(".")[0] + "_report.txt", "wt") as f:
        f.write(new_report)


if __name__ == "__main__":
    main()
