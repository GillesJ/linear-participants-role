from imblearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import f_classif, SelectPercentile, mutual_info_classif
from sklearn.model_selection import GridSearchCV
import datahandler as dh
import settings as s
import util
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection
from imblearn.combine import SMOTETomek
import pipeline
import numpy as np
from featureselect import bns
import cPickle as pickle

# c = pickle.load(open('/home/gilles/repos/cbrole/static/TEST149047537747_cbevent_en/all_pipeline_cv_results.pkl', 'rb'))
# util.ensure_dir(s.OPT_DIRP)

# X, y = dh.load_data(s.DATA_FP, n_features=s.NUM_FEATURES, memmapped=False)
# X, y = X[0:100, :], y[0:100]

#
# linearsvc_params = {'loss': ['squared_hinge'], 'C': [0.02, 0.2, 2, 20, 200], 'class_weight': ['balanced', None]}
# nusvc_params = {'nu': [0.002, 0.02], 'kernel': ['rbf'], 'gamma': ['auto', 0.002, 0.02, 0.2, 2, 20, 200], 'class_weight': ['balanced', None], }
# scaler_params = {}
# randomus_params = {}
# featselect_params = {'percentile': [33, 67]}
# logregr_params = {'C': [0.02, 0.2, 2, 20, 200], 'class_weight': ['balanced', None]}
# sgd_params = {'loss': ['modified_huber', 'perceptron'], 'n_iter': [np.ceil(10**6 / s.NUM_INST)], 'class_weight': ['balanced', None]} # Empirically, we found that SGD converges after observing approx. 10^6 training samples. Thus, a reasonable first guess for the number of iterations is n_iter = np.ceil(10**6 / n), where n is the size of the training set.
# passaggr_params = {'loss': ['hinge', 'squared_hinge'], 'C': [0.02, 0.2, 2, 20, 200], 'n_iter': [np.ceil(10**6 / s.NUM_INST)], 'class_weight': ['balanced', None]}
#
# featselect_params = {'percentile': [33, 67]}
# randomus = {}
# smotetomek_params = {}
#
# pipe_steps = [(
#     'featselect', [
#         (None, None),
#         (SelectPercentile(f_classif), featselect_params),
#         (SelectPercentile(mutual_info_classif), featselect_params),
#     ]), (
#     'resample', [
#         (None, None),
#         (RandomUnderSampler(), randomus_params),
#         (SMOTETomek(), smotetomek_params),
#     ]), (
#     'classify', [
#         [(MaxAbsScaler(), scaler_params), (LinearSVC(), linearsvc_params)],
#         [(MaxAbsScaler(), scaler_params), (NuSVC(), nusvc_params)],
#         (LogisticRegression(), logregr_params),
#     ]),
# ]
#
# steps, param_grids = pipeline.make_pipelines(pipe_steps, alt_order=['resample','featselect', 'classify'])
# all_results  = {}
# for (steps, param_grid) in zip(steps, param_grids):
#     pipe = Pipeline(steps)
#     grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=pipeline.my_scorer, n_jobs=s.CV_N_JOBS, cv=s.CV,
#                                verbose=10, error_score=0, return_train_score=False)
#     grid_search.fit(X, y)
#     all_results.update(grid_search.cv_results_)

# make a generator which makes a list of all possible pipelines with their param to be grid searched


# TEXT MESSAGE TEST

import sklearn.datasets
import numpy as np
import random
from sklearn.metrics import classification_report

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]


from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

paramgrid = {
    "kernel": ["rbf"],
    "C": np.logspace(-9, 9, num=25, base=10),
    "gamma": np.logspace(-9, 9, num=25, base=10),
}

random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV

cv = EvolutionaryAlgorithmSearchCV(
    estimator=SVC(),
    params=paramgrid,
    scoring="f1",
    cv=StratifiedKFold(n_splits=4),
    verbose=1,
    population_size=50,
    gene_mutation_prob=0.10,
    gene_crossover_prob=0.5,
    tournament_size=3,
    generations_number=5,
    n_jobs=4,
)
cv.fit(X, y)
