"""
Settings file unique to each experiment run. Specifies search technique, parameters, dataset filepath, etc.
"""
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.svm import LinearSVC, SVC, NuSVC
from collections import OrderedDict
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from time import time
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import util
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    ExtraTreesClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)
import datahandler
from featureselect import bns
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection
from custom_classifiers import CascadingClassifier
from imblearn.combine import SMOTETomek
import os
import numpy as np

TIMESTAMP = str(time()).replace(".", "")
RANDOM_STATE = 42

# English specific settings
langspec = {
    "en": {
        "NUM_FEATURES": 871044,
        "NUM_INSTANCES": 113694,
        "FEATURE_GROUPS": {
            "a_topicmodels": ["lda", "lsi"],
            "b_lexica": [
                "names",
                "allness",
                "diminishers",
                "intensifiers",
                "negations",
                "imperative",
                "person_alternation",
                "ass",
                "profanity",
            ],
            "c_subjectivity": [
                "smiley",
                "liwc",
                "afinn",
                "geninq",
                "liu",
                "mpqa",
                "msol",
            ],
            "d_chargram": ["tch2gr", "tch3gr", "tch4gr"],
            "e_wordgram": ["w1gr", "w2gr", "w3gr"],
        },
        "MAPDICT_FP": "/home/gilles/data/cbvectordata/en_cb3/cb3_en_REDO.mapdict",
        "DATA_FP": "/home/gilles/data/cbvectordata/en_cb3/cb3_en_REDO_3_roles.svm",
    },
    "nl": {
        "NUM_FEATURES": 795072,
        "NUM_INSTANCES": 78387,
        "FEATURE_GROUPS": {},
        "MAPDICT_FP": "/home/gilles/data/cbvectordata/nl_cb3/cb3_nl.mapdict",
        "DATA_FP": "/home/gilles/data/cbvectordata/nl_cb3/cb3_nl_3_roles.svm",
    },
    "en_detect": {
        "NUM_FEATURES": 871044,
        "NUM_INSTANCES": 113694,
        "FEATURE_GROUPS": {
            "a_topicmodels": ["lda", "lsi"],
            "b_lexica": [
                "names",
                "allness",
                "diminishers",
                "intensifiers",
                "negations",
                "imperative",
                "person_alternation",
                "ass",
                "profanity",
            ],
            "c_subjectivity": [
                "smiley",
                "liwc",
                "afinn",
                "geninq",
                "liu",
                "mpqa",
                "msol",
            ],
            "d_chargram": ["tch2gr", "tch3gr", "tch4gr"],
            "e_wordgram": ["w1gr", "w2gr", "w3gr"],
        },
        "MAPDICT_FP": "/home/gilles/data/cbvectordata/en_cb3/cb3_en_REDO.mapdict",
        "DATA_FP": "/home/gilles/data/cbvectordata/en_cb3/cb3_en_REDO_bully_event.svm",
    },
    "nl_detect": {
        "NUM_FEATURES": 795072,
        "NUM_INSTANCES": 78387,
        "FEATURE_GROUPS": {},
        "MAPDICT_FP": "/home/gilles/data/cbvectordata/nl_cb3/cb3_nl.mapdict",
        "DATA_FP": "/home/gilles/data/cbvectordata/nl_cb3/cb3_nl_bully_event.svm",
    },
}
# input/output settings
LANGUAGE = "nl"
DATA_FP = langspec[LANGUAGE]["DATA_FP"]
RUN_ID = "{}_VOTING_3CBROLE_{}".format(LANGUAGE, TIMESTAMP)
MAPDICT_FP = langspec[LANGUAGE]["MAPDICT_FP"]
DATASET_NAME = "3CBROLE_{}".format(LANGUAGE.upper())
NUM_FEATURES = langspec[LANGUAGE]["NUM_FEATURES"]
NUM_INST = langspec[LANGUAGE]["NUM_INSTANCES"]
ABLATION = False
OPT_DIRP = "/home/gilles/repos/cbrole/static/{}/".format(RUN_ID)
FEATURE_GROUPS = langspec[LANGUAGE]["FEATURE_GROUPS"]
MULTICLASS = True  # set to False for binary classification and true for multiclass
HOLDOUT = 0.1
LABELMAP = dict(((-1, "no_bullying"), (0, "harasser"), (1, "victim"), (2, "defender")))
PARTIALRUN = False  # "/home/gilles/repos/cbrole/static/BASIC_nl_3_CBROLE_150401763413/partialruninfo.json"

# CROSSVALIDATION search settings
k_folds = 5
CV = StratifiedKFold(n_splits=k_folds, shuffle=False, random_state=RANDOM_STATE)
CV_N_JOBS = 15
# SEARCHCV = GridSearchCV # search strat to be used in crossvalidation can be GridSearchCV() or RandomSearchCV() TODO allow this to be set
# Scorer settings called in myscorer of pipeline.py
SCORER_METRIC = "f1"
SCORE_AVERAGING = "macro"
SCORER_FOLD_LOG_DIRP = os.path.join(OPT_DIRP, "fold_log")
SCORER_FOLD_MODEL_DIRP = os.path.join(OPT_DIRP, "fold_models")

# For the voting clf

votingclf = {
    "en": VotingClassifier(
        estimators=[
            (
                "LogisticRegression",
                LogisticRegression(
                    C=0.2, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
            (
                "LinearSVC",
                LinearSVC(
                    C=0.02,
                    loss="squared_hinge",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "PassiveAggressiveClassifier",
                PassiveAggressiveClassifier(
                    C=0.02,
                    loss="hinge",
                    class_weight=None,
                    n_iter=9.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        voting="hard",
    ),
    "nl": VotingClassifier(
        estimators=[
            (
                "logres",
                LogisticRegression(
                    C=0.2, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
            (
                "passaggr",
                PassiveAggressiveClassifier(
                    C=0.02,
                    loss="squared_hinge",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "linearsvc",
                LinearSVC(
                    C=0.02,
                    loss="squared_hinge",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        voting="hard",
    ),
}

# PARAM_GRID defines all transformators and ends with a classifier. Can take embedded pipelines. Embedded pipelines are
# especially handy for scaling. Some classifiers require scaled data while others do not.
scaler_params = {}

featselect_params = {"percentile": [33, 67]}

randomus_params = {"ratio": [0.01]}
# randomus_params = {'ratio': 'auto'}
onesidedsel_params = {}
smotetomek_params = {"ratio": ["auto", 0.5]}

linearsvc_params = {
    "loss": ["squared_hinge"],
    "C": [0.02, 0.2, 1, 2, 20, 200],
    "class_weight": ["balanced", None],
}
nusvc_params = {
    "nu": [0.002, 0.02],
    "kernel": ["rbf"],
    "gamma": ["auto", 0.002, 0.02, 0.2, 1, 2, 20, 200],
    "class_weight": ["balanced", None],
}
logregr_params = {"C": [0.02, 0.2, 1, 2, 20, 200], "class_weight": ["balanced", None]}
ridge_params = {
    "alpha": [0.05, 0.5, 1, 5, 50, 500],
    "solver": ["sag"],
    "class_weight": ["balanced", None],
}  # Alpha corresponds to C^-1 in other linear models such as LogisticRegression or LinearSVC.
sgd_params = {
    "loss": ["modified_huber", "perceptron"],
    "n_iter": [np.ceil(float(10 ** 6) / NUM_INST)],
    "class_weight": ["balanced", None],
}  # Empirically, we found that SGD converges after observing approx. 10^6 training samples. Thus, a reasonable first guess for the number of iterations is n_iter = np.ceil(10**6 / n), where n is the size of the training set.
passaggr_params = {
    "loss": ["hinge", "squared_hinge"],
    "C": [0.02, 0.2, 1, 2, 20, 200],
    "class_weight": ["balanced", None],
}
dectree_params = {
    "criterion": ["gini", "entropy"],
    "max_features": [None, "sqrt"],
    "class_weight": ["balanced", None],
}

# adaboost_params = {} # META-ESTIMATOR TO BE SET AS WINNER FROM PREVIOUS RUN
extratree_params = {
    "n_estimators": [500, 1000],
    "criterion": ["gini", "entropy"],
    "max_features": [None, "sqrt"],
    "class_weight": ["balanced", None],
}
gradboost_params = {
    "loss": ["deviance", "exponential"],
    "max_depth": [3, 30, 300],
    "n_estimators": [500, 1000],
    "max_features": [None, "sqrt"],
}
rf_params = {
    "n_estimators": [500, 1000],
    "criterion": ["gini", "entropy"],
    "max_features": [None, "sqrt"],
    "class_weight": ["balanced", None],
}

knn_params = {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]}
radn_params = {"radius": [1.0, 1.5, 2.0], "weights": ["uniform", "distance"]}

# isof_params = {'n_estimators': [500, 1000], 'contamination': float(NUM_NEG)/NUM_POS, 'max_features': [None, 'sqrt']}
# elenv_params = {'contamination': float(NUM_NEG)/NUM_POS}

voting_params = {}

# CASCADE ENSEMBLE APPROACH
# threestage_cascading_cf = {
#     'en':
#         CascadingClassifier(
#             [
#                 ('a', LinearSVC()),
#                 ('b', LinearSVC()),
#                 ('c', LinearSVC()),
#             ],
#             {
#             'a__nobully': -1,
#             'a__bully': {
#                 'b__harasser': 0,
#                 'b__notharasser': {
#                     'c__victim': 1,
#                     'c__defender': 2,
#             }}}
#         ),
# }

twostage_cascade = {
    "en": CascadingClassifier(
        [("a", LinearSVC()), ("b", votingclf["en"]),],
        {
            "a__nobully": -1,
            "a__bully": {"b__harasser": 0, "b__defender": 1, "b__bystander": 2,},
        },
    ),
    "nl": CascadingClassifier(
        [("a", LinearSVC()), ("b", LogisticRegression()),],
        {
            "a__nobully": -1,
            "a__bully": {"b__harasser": 0, "b__defender": 1, "b__bystander": 2,},
        },
    ),
}
twostage_cascade_params = {
    "en": {
        "a__C": [0.02, 0.2, 2, 20, 200],
        "a__class_weight": ["balanced"],
        "a__loss": ["squared_hinge"],
    },
    "nl": {
        "a__C": [0.02, 0.2, 2, 20, 200],
        "a__class_weight": ["balanced"],
        "a__loss": ["squared_hinge"],
        "b__C": [0.02, 0.2, 1, 2, 20, 200],
        "b__class_weight": ["balanced", None],
    },
}


PIPE_STEPS = [
    (
        "featselect",
        [
            (None, None),
            (SelectPercentile(f_classif), featselect_params),
            (SelectPercentile(mutual_info_classif), featselect_params),
        ],
    ),
    (
        "resample",
        [
            (None, None),
            (RandomUnderSampler(random_state=RANDOM_STATE), randomus_params),
            # (SMOTETomek(random_state=RANDOM_STATE), smotetomek_params),
        ],
    ),
    (
        "classify",
        [
            [
                (MaxAbsScaler(), scaler_params),
                (twostage_cascade[LANGUAGE], twostage_cascade_params[LANGUAGE]),
            ]
            # [(MaxAbsScaler(), scaler_params), (LinearSVC(random_state=RANDOM_STATE), linearsvc_params)], # PRETTY FAST
            # (NuSVC(random_state=RANDOM_STATE), nusvc_params), # TOO SLOW
            # [(MaxAbsScaler(), scaler_params), (LogisticRegression(random_state=RANDOM_STATE), logregr_params)], # FAST
            # [(MaxAbsScaler(), scaler_params), (SGDClassifier(random_state=RANDOM_STATE), sgd_params)], # FAST
            # [(MaxAbsScaler(), scaler_params), (PassiveAggressiveClassifier(random_state=RANDOM_STATE), passaggr_params)], # FAST
            # (RidgeClassifier(random_state=RANDOM_STATE), ridge_params), # PRETTY FAST TOO MUCH MEMORY
            # (DecisionTreeClassifier(random_state=RANDOM_STATE), dectree_params), # FAST
            # # ENSEMBLE classifiers
            # (AdaBoostClassifier(random_state=RANDOM_STATE), adaboost_params), # USE WITH WINNER
            # (ExtraTreesClassifier(random_state=RANDOM_STATE), extratree_params), # TOO SLOW
            # (GradientBoostingClassifier(random_state=RANDOM_STATE), gradboost_params), # TOO SLOW
            # (RandomForestClassifier(random_state=RANDOM_STATE), rf_params), # TOO SLOW
            # # NEAREST NEIGHBOR classifiers
            # (KNeighborsClassifier(), knn_params), # PRETTY FAST
            # (RadiusNeighborsClassifier(), radn_params) # radius is fickle
            # # OUTLIER classifiers FLIP -1, 1 labels for this
            # (IsolationForest(random_state=RANDOM_STATE), isof_params),
            # (EllipticEnvelope(random_state=RANDOM_STATE), elenv_params),
            # # VOTING CLF
            # [(MaxAbsScaler(), scaler_params),  (votingclf[LANGUAGE], voting_params)],
        ],
    ),
]
ALT_ORDER = ["resample", "featselect", "classify"]
