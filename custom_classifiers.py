#!/usr/bin/env python
"""
custom_classifiers.py
cbrole 
10/5/17
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import numpy as np
import util

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy import sparse

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from treelib import Tree


class CascadingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Cascading classifier approach in which.
    - to give a series of classifier with OR without filter-label.
    - to have the option to append previous stage prediction output, or to use only the prediction.
    - X should always have same shape, y can be different: solve  with labelmap
    - conditions each step must result in one final label
    """

    def __init__(self, estimators, label_map):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.stages = [t[0] for t in estimators]
        self.label_map = label_map
        self.named_label_map_ = dict(label_map)

    def check_labelmap_(self, y):
        #  further checks on the label map: label mapping should
        # check if labelmap has mapping for each step, check if each steps mapping is complete
        # check in the order of named_estimators
        # each step must end in one of the original ys  final labels
        # chekc no overlap in names and labels
        all_stages = self.named_estimators.keys()
        labels_in_stage = {}
        for name, labels in self.named_label_map_.iteritems():
            name_labels = name.split("_")
            stage_name = name_labels[0]
            if stage_name not in all_stages:
                raise AttributeError(
                    "Invalid label map. Stage name does not correspond with named classifiers."
                )
            # TO DO  FINISH THIS

    def encode_labelmap_(self, labelencoder):
        def encode_walk(node, encoded_labelmap, labelencoder):
            for key, item in node.iteritems():
                key_enc = labelencoder.transform([key])[0]
                if isinstance(item, dict):
                    encoded_labelmap[key_enc] = encode_walk(item, {}, labelencoder)
                else:
                    item_enc = labelencoder.transform([item])[0]
                    encoded_labelmap[key_enc] = item_enc
            return encoded_labelmap

        return encode_walk(self.named_label_map_, {}, labelencoder)

    def stage_walk_(self, node, stagename, in_stage):
        for key, item in node.iteritems():
            if stagename in key.split("__") and (key, item) not in in_stage:
                in_stage.append((key, item))
            if isinstance(item, dict):
                in_stage = self.stage_walk_(item, stagename, in_stage)
        return in_stage

    def map_labels_(self, y, stagename):
        """
        The
        :param y:
        :param stagename:
        :return:
        """
        classes = np.unique(y)
        stage = self.stage_walk_(self.label_map, stagename, [])

        replace_map = dict(
            (
                self.le_casc_.transform([node])[0],
                self.le_casc_.transform([name_label])[0],
            )
            if not isinstance(node, dict)
            else ("rest", self.le_casc_.transform([name_label])[0])
            for name_label, node in stage
        )

        if "rest" in replace_map:
            rest = dict(
                (lab, replace_map["rest"])
                for lab in [
                    lab
                    for lab in classes
                    if lab not in replace_map.keys() and lab != "rest"
                ]
            )
            replace_map.pop("rest", None)
            replace_map.update(rest)

        return replace_map

    def apply_mapping_(self, y, map):

        if not isinstance(y, np.ndarray):
            raise TypeError("y is not a numpy array.")

        return np.vectorize(map.__getitem__)(y)

    def slice_Xy_(self, X, y, stagename):
        # get previous stage to current stagename
        y_classes = self.le_casc_.inverse_transform(np.unique(y)).astype(int)
        remove_label = [
            stagename_label
            for stagename_label, nextstage in self.stage_walk_(
                self.label_map, stagename, []
            )
            if nextstage in y_classes
        ]
        remove_label = [
            list(util.gen_dict_extract(remove_lab, self.label_map))[0]
            for remove_lab in remove_label
        ]
        remove_label = self.le_casc_.transform(remove_label)
        keep_idc = []
        for remove_lab in remove_label:
            keep_idc.append(np.where(y != remove_lab))  # make the slice
        keep_idc = np.concatenate(*keep_idc)
        keep_idc.sort()

        X = X[
            keep_idc
        ]  # TODO do not slice and copy but drop colums in X to save memory
        y = y[keep_idc]

        return X, y, keep_idc

    def update_y_(self, y, name):
        """
        Set the predictions made in current stage so that the prediction can be passed to the next stage.
        :param y:
        :param pred:
        :return:
        """
        pred = self.stage_pred_[name]
        classes = self.le_casc_.inverse_transform(np.unique(y)).astype(int)
        y_leafs = [
            (stagename_label, map)
            for stagename_label, map in self.stage_walk_(self.label_map, name, [])
            if map in classes
        ]

        # replace all y that with the leaf value if they have the leaf value in the prediction
        for stagename_label, leaf in y_leafs:
            stage_lab = self.le_casc_.transform([stagename_label])[0]
            leaf_pred_idc = np.where(pred == stage_lab)
            y[leaf_pred_idc] = self.le_casc_.transform([leaf])[0]
        return y

    def get_map_labels_(self, label_map):
        def flatten_node(node, flat):
            if not isinstance(flat, list):
                raise TypeError("Collector should be list.")
            for item, val in node.iteritems():
                flat.append(item)
                if isinstance(val, dict):
                    flat.extend(flatten_node(val, []))
                else:
                    flat.append(val)

            return flat

        map_labels = flatten_node(label_map, [])

        return map_labels

    def fit(self, X, y):
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError(
                "Invalid estimator. Pass a list of (estimator_name, sklearn-estimator) tuples."
            )

        if (
            self.label_map is None
            or len(self.label_map) == 0
            or len(self.named_label_map_) < 1
        ):
            raise AttributeError("Invalid label map.")

        self.le_ = LabelEncoder()  # this block is arguably not needed
        self.le_.fit(y)
        self.classes_ = self.le_.classes_

        # we use a labeleconding approach to work with labels nicely
        self.le_casc_ = LabelEncoder()
        self.le_casc_.fit(self.get_map_labels_(self.label_map))
        self.casc_classes_ = self.le_casc_.classes_
        self.named_label_map_ = self.encode_labelmap_(self.le_casc_)
        # self.check_labelmap_(self.le_.transform(y))  # check if mapping is valid

        self.fitted_estimators_ = []
        self.instance_map_ = {}
        self.y_mapping = []
        self.stage_pred_ = {}

        y_casc = self.le_casc_.transform(
            y
        )  # making a copy of y is not expensive and good for debugging, we will not do the same with X

        for stage_name, clf in self.estimators:
            # IMPORTANT NOTES:
            # - y_casc always maintains its final prediction labels!
            # - each stage hasto have at least 1 final label
            # FIT()
            # step 1: map original y labels to the ones in the stage, keep mapping
            # -  function to make mapping dict per stage and  save these mappings
            # -  function to apply mapping producing y_mapped on which training is performed
            # step 2: train
            # step 3: predict y at stage for slicing
            # step 4: change the y_casc so that final labels in prediction are the same (percolate predictions)
            # step 5: slice X and yso  that only non-final labels are kept in, produce a slice map:
            # slice_map: [(stage_name, {label: idc}), ]

            # map the label classes for classification in current stage
            # print("{}: X has shape {}\ny_casc has shape  {}\ny_casc has labels {}".format(stage_name, X.shape, y_casc.shape, np.unique(y_casc, return_counts=True)))
            y_map = self.map_labels_(y_casc, stage_name)
            self.y_mapping.append((stage_name, y_map))
            y_mapped = self.apply_mapping_(y_casc, y_map)
            # fit the estimator
            fitted_clf = clone(clf).fit(X, y_mapped)
            self.fitted_estimators_.append((stage_name, fitted_clf))
            # predict for updating of y followed by slicing of instances
            self.stage_pred_[stage_name] = fitted_clf.predict(X)
            # print("Classes y_casc before prediction update: {}".format(np.unique(y_casc, return_counts=True)))
            y_casc = self.update_y_(
                y_casc, stage_name
            )  # update y with the preds as ground truth
            # print("Classes y_casc after prediction update: {}".format(np.unique(y_casc, return_counts=True)))

            if stage_name != self.estimators[-1][0]:  # do not slice in final step
                X, y_casc, idc_slice = self.slice_Xy_(
                    X, y_casc, stage_name
                )  # slice the instances and labels based on previous stage prediction
                self.instance_map_[stage_name] = idc_slice
        return self

    def predict(self, X):
        return self.predict_(X)

    def predict_(self, X):

        predictions = []
        for stage_name, clf in self.fitted_estimators_:

            keep_idc = []
            y_pred_stage = clf.predict(X)
            # slice X unless final stage
            if stage_name != self.fitted_estimators_[-1][0]:
                remove_label = [
                    stagename_label
                    for stagename_label, nextstage in self.stage_walk_(
                        self.label_map, stage_name, []
                    )
                    if not isinstance(nextstage, dict)
                ]
                remove_label = self.le_casc_.transform(remove_label)
                for remove_lab in remove_label:
                    # keep_indc = np.where(y_pred_stage != remove_lab)[0].tolist()
                    keep_idc.extend(
                        np.where(y_pred_stage != remove_lab)[0].tolist()
                    )  # make the slice
                keep_idc.sort()
                X = X[keep_idc]

            predictions.append((stage_name, y_pred_stage, keep_idc))

        y_predictions = []
        for stage_name, pred, keep_idc in predictions:
            y_mapping = dict(self.y_mapping)[stage_name]
            y_rev_mapping = {}
            non_leaf = set()
            for k, v in y_mapping.iteritems():
                if v not in y_rev_mapping:
                    y_rev_mapping[v] = k
                else:
                    non_leaf.add(v)
            for non_lf in set(non_leaf):
                y_rev_mapping[non_lf] = non_lf
            y_predictions.append(
                (stage_name, np.vectorize(y_rev_mapping.__getitem__)(pred), keep_idc)
            )

        # reconstruct predictions
        predictions_rvrs = list(
            reversed(
                [
                    (_, np.copy(y_pred_stage), __)
                    for (_, y_pred_stage, __) in y_predictions
                ]
            )
        )
        # loop through in reverse order and replace by keep_idc
        for i, (stage_name, pred, keep_idc) in enumerate(predictions_rvrs):
            nxt_i = i + 1
            if nxt_i < len(predictions_rvrs):
                stage_name_nxt, pred_nxt, keep_idc_nxt = predictions_rvrs[nxt_i]
                for idx, val in zip(keep_idc_nxt, pred):
                    pred_nxt[idx] = val
            else:
                break

        y_pred = predictions_rvrs[-1][1]
        return self.le_casc_.inverse_transform(y_pred).astype(int)

    def get_params(self, deep=True):
        if not deep:
            return super(CascadingClassifier, self).get_params(deep=False)
        else:
            out = super(CascadingClassifier, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            out.update(self.named_label_map_.copy())
            return out


class StackingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    StackingClassifier approach in which a metaclassifier is trained on the prediction outputs of multiple base-estimators.
    The training method is implemented by calling the fit methods of the underlying estimators, concatenate the
    predictions to the input features and fit our classifier to the new data.
    Source:
    http://algoadventures.com/sklearn-from-the-source-code-up-basics/
    http://algoadventures.com/hacking-sklearn-building-your-own-estimator-part-1/
    http://algoadventures.com/hacking-sklearn-building-your-own-estimator-part-2/
    """

    def __init__(self, estimators, classifier):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.classifier = classifier
        self.named_classifier = dict(classifier)

    def fit(self, X, y):
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError("Invalid estimator.")

        if (
            self.classifier is None
            or len(self.classifier) == 0
            or len(self.named_classifier) > 1
        ):
            raise AttributeError("Invalid classifier.")

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.fitted_estimators_ = []
        for name, clf in self.estimators:
            fitted_clf = clone(clf).fit(X, self.le_.transform(y))
            self.fitted_estimators_.append(fitted_clf)
        ensemblePredictions = self.predict_(X)
        self.classifier_ = clone(self.classifier[0][1]).fit(
            np.concatenate((X, ensemblePredictions), axis=1), self.le_.transform(y)
        )
        return self

    def predict(self, X):
        ensemblePredictions = self.predict_(X)
        return self.classifier_.predict(
            np.concatenate((X, ensemblePredictions), axis=1)
        )

    def predict_(self, X):
        predictions = []
        for clf in self.fitted_estimators_:
            predictions.append(clf.predict(X))
        return np.column_stack(predictions)

    def get_params(self, deep=True):
        if not deep:
            return super(StackingClassifier, self).get_params(deep=False)
        else:
            out = super(StackingClassifier, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            out.update(self.named_classifier.copy())
            return out


if __name__ == "__main__":

    # X, y = make_classification(n_samples=10000, n_classes=5, n_features=100, n_redundant=2, n_informative=15,
    #                            random_state=1, n_clusters_per_class=1)
    #
    # clfstages = [
    #     ('adetect', LinearSVC()),
    #     ('bdetect', DecisionTreeClassifier()),
    #     ('cmulti', GaussianNB()),
    # ]
    #
    # labelmap  = {
    #     'adetect__no': 0,
    #     'adetect__yes': {'bdetect__no': 1,
    #            'bdetect__yes': {
    #                'cmulti__a': 2,
    #                'cmulti__b': 3,
    #                'cmulti__c': 4}}
    # }  # if clf a predicts 0, final is label zero, if clf a predict 1: feed to clf b

    X, y = make_classification(
        n_samples=10000,
        n_classes=4,
        n_features=100,
        n_redundant=2,
        n_informative=15,
        random_state=1,
        n_clusters_per_class=1,
    )
    y = y - 1
    print (np.unique(y, return_counts=True))

    clfstages = [
        ("a", LinearSVC()),
        ("b", LinearSVC()),
    ]
    labelmap = {
        "a__notbully": -1,
        "a__bully": {"b__harasser": 0, "b__victim": 1, "b__bystander": 2,},
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=40
    )
    clf = CascadingClassifier(clfstages, labelmap)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print (classification_report(y_test, y_pred))

    parameters = {"a__C": [1, 2, 20], "b__C": [1, 2, 20, 200]}
    gs = GridSearchCV(clf, parameters, scoring="f1_macro", verbose=2)
    gs.fit(X, y)
    y_pred = gs.predict(X_test)
    print (classification_report(y_test, y_pred))
    print gs.best_params_

    clf = LinearSVC(C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print (classification_report(y_test, y_pred))

    # classifiers = [
    #     ('NB', GaussianNB()),
    #     ('DTC', DecisionTreeClassifier())]
    # clf = StackingClassifier(classifiers, [('RF', RandomForestClassifier(max_depth=5, max_features=1))])
    # parameters = {"DTC__max_features": [1, 5, 10, 15], "RF__n_estimators": [5, 8, 10]}
    # gs = GridSearchCV(clf, parameters, scoring="f1", verbose=2)
    # gs.fit(X, y)
    # print gs.best_params_
    #
