from sklearn.externals.joblib import load
import settings as s
import datahandler as dh
import json
import os
import numpy_indexed as npi
import numpy as np
from functools import reduce
import reporter
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import Normalizer, MaxAbsScaler
import util


def reconstruct_list(idc_mappings, filter_label=[]):
    """
    Reconstruct a numpy array from a list value-index mappings. The list will be updated in order.
    :param idc_mappings: a list of mapping dict {item value: indices with value}
    :return: a reconstructed list from the mappings
    """

    if filter_label:  # normalize idc
        assert len(filter_label) == len(idc_mappings) - 1
        for i, filt_label in enumerate(filter_label):
            assert sum(len(v) for v in idc_mappings[i + 1].itervalues()) == len(
                idc_mappings[i][filt_label]
            )
            for next_label, next_idc in idc_mappings[i + 1].iteritems():
                true_idc = np.array(idc_mappings[i][filt_label])[np.array(next_idc)]
                idc_mappings[i + 1][next_label] = true_idc

    max_idc = max(max(idc) for idc in idc_mappings[0].values())
    result = np.empty([max_idc + 1,], dtype=int)
    for map in idc_mappings:
        for value, idc in map.iteritems():
            for idx in idc:
                result[idx] = value

    return result.astype(int)


def cascade_predict(X_out, detect, role, detect_label):

    # # refit both detect and role
    # detectparams = detect.best_estimator_.get_params()
    # roleparams=  role.best_estimator_.get_params()
    # re_detect = detect.best_estimator_.set_params(**detectparams)
    # re_role = role.best_estimator_.set_params(**roleparams)
    #
    # detect = re_detect.fit(X_in, y_in)
    # role = re_role.fit(X_in, y_in)

    # predict holdout for detection y_pred_detect
    y_pred_dtct = detect.predict(X_out).astype(int)

    # split off positive and negs from y and X in y_pred_detect_minus1, y_pred_detect_1
    unique_dtct, idx_groups_dtct = npi.group_by(
        y_pred_dtct, np.arange(len(y_pred_dtct))
    )
    pred_dtct_idc = dict(zip(unique_dtct, idx_groups_dtct))
    dtct_idc = pred_dtct_idc[detect_label]
    X_dtct = X_out[dtct_idc]

    # predict X_1: return y_pred_1_pred
    y_pred_role = role.predict(X_dtct).astype(int)
    unique_role, idx_groups_role = npi.group_by(
        y_pred_role, np.arange(len(y_pred_role))
    )
    pred_role_idc = dict(zip(unique_role, idx_groups_role))

    # reconstruct full y_pred
    y_pred = reconstruct_list(
        [pred_dtct_idc, pred_role_idc], filter_label=[detect_label]
    )

    return y_pred


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


if __name__ == "__main__":

    LANGUAGE = "nl"
    CASCADE = False
    BOOTSTRAP = True
    CONFUSION_MATRIX = True

    # _, _ = dh.load_data(DATA_FP, n_features=s.langspec[LANGUAGE]['NUM_FEATURES'], memmapped=False)
    # _, _ = dh.load_data(DATA_FP, n_features=s.langspec[LANGUAGE]['NUM_FEATURES'], memmapped=False)

    langspec = {
        "en": {
            "detect_modelfp":
            # "/home/gilles/repos/cbrole/static/DETECT_LSVC_en_detect_3_CBROLE_150428295005/cv_pipelines/randomundersampler+mutual_info_classif+maxabsscaler+linearsvc/150428295005_grid_search.joblibpkl",
            "/home/gilles/repos/cbrole/static/DETECT_LSVC_en_detect_3_CBROLE_150428295005/cv_pipelines/f_classif+maxabsscaler+linearsvc/150428295005_grid_search.joblibpkl",  # the alternative
            "role_modelfp":
            # "/home/gilles/repos/cbrole/static/VOTING_en_3_CBROLE_150426243867/cv_pipelines/maxabsscaler+votingclassifier/150426243867_grid_search.joblibpkl",
            "/home/gilles/repos/cbrole/static/VOTING_en_3_CBROLE_150426243867/cv_pipelines/mutual_info_classif+maxabsscaler+votingclassifier/150426243867_grid_search.joblibpkl",  # the alternative
        },
        "nl": {
            "detect_modelfp": "/home/gilles/repos/cbrole/static/LINEARSVC_3_CBEVENT149968157182_nl/cv_pipelines/f_classif+linearsvc/149968157182_grid_search.joblibpkl",
            "role_modelfp": "/home/gilles/repos/cbrole/static/LINEARSVC_3_CBROLE149744106886_cbrole_nl/cv_pipelines/f_classif+linearsvc/149744106886_grid_search.joblibpkl",
        },
    }
    METRIC = ["fscore", "precision", "recall", "acc"]
    detect_label = 1

    util.ensure_dir("/home/gilles/repos/cbrole/static/CASCADE_{}".format(LANGUAGE))
    # load heldout X, y
    DATA_FP = s.langspec[LANGUAGE]["DATA_FP"]
    # X, y = dh.load_data(DATA_FP, n_features=s.langspec[LANGUAGE]['NUM_FEATURES'], memmapped=False)

    run_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(langspec[LANGUAGE]["role_modelfp"]))
    )
    NUM_FEATURES_POSTSPLIT = json.load(
        open(os.path.join(run_dir, "holdinout_split_indices.json"), "rt")
    )["num_features"]
    X_in, y_in = dh.load_data(
        "{}/holdin.svm".format(run_dir),
        n_features=NUM_FEATURES_POSTSPLIT,
        memmapped=False,
    )
    X_out, y_out = dh.load_data(
        "{}/holdout.svm".format(run_dir),
        n_features=NUM_FEATURES_POSTSPLIT,
        memmapped=False,
    )

    # load detection model
    detect_fp = langspec[LANGUAGE]["detect_modelfp"]
    role_fp = langspec[LANGUAGE]["role_modelfp"]
    detect = load(detect_fp)
    role = load(role_fp)
    all_classes = reduce(np.union1d, (detect.classes_, role.classes_))

    # y_out_pred = cascade_predict(X_in, y_in, X_out, detect, role, detect_label)
    y_out_pred = cascade_predict(X_out, detect, role, detect_label)

    # BOOTSTRAP RESAMP
    if BOOTSTRAP:
        sample_size = y_out.shape[0]
        n = 10000
        bootstrapresample_scores = reporter.bootstrap_resample(
            y_out, y_out_pred, n, sample_size
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

        print("holdout score BS")
        for k, v in bootstrap_score.iteritems():
            if "_avg" in k:
                print("{}: {}".format(k, np.round(v, decimals=4) * 100))

        for metric in METRIC:
            plot_fp = "/home/gilles/repos/cbrole/static/CASCADE_{}/cascade_{}_hist.png".format(
                LANGUAGE, metric
            )
            reporter.plot_histogram(bootstrap_score[metric], plot_fp)

    out_scores = reporter.calculate_scores(y_out, y_out_pred)

    print("Holdout score non-BS:\n{}".format(out_scores))

    # Confusion matrix
    if CONFUSION_MATRIX:
        cm_plot_fp = "/home/gilles/repos/cbrole/static/CASCADE_{}/cascade_confusion_matrix.png".format(
            LANGUAGE
        )
        reporter.savefig_confusion_matrix(y_out, y_out_pred, cm_plot_fp)

    # IN CROSSVALIDATION
    cv = s.CV
    folds = cv.split(X_in, y_in)

    all_scores = []
    baseline_majority_scores = []
    baseline_random_scores = []
    for (i, (train_idc, test_idc)) in enumerate(folds):
        print("Fold {}:".format(i))
        X_train = X_in[train_idc]
        y_train = y_in[train_idc]
        X_test = X_in[test_idc]
        y_test = y_in[test_idc]

        bl_maj = reporter.majority_baseline(y_test)
        baseline_majority_scores.append(bl_maj)
        bl_rand = reporter.random_baseline(y_test)
        baseline_random_scores.append(bl_rand)

        y_pred = cascade_predict(X_test, detect, role, detect_label)

        scores = reporter.calculate_scores(y_test, y_pred)

        print(scores)
        all_scores.append(scores)

    print("CV scores")
    print_overview(all_scores)
    print("\nCV majority baseline")
    print_overview(baseline_majority_scores)
    print("\nCV random baseline")
    print_overview(baseline_random_scores)


# import keras
# import keras.backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
# import numpy as np
# from sklearn.metrics import roc_auc_score
# import settings as s
# import datahandler as dh
#
# X, y = dh.load_data(s.DATA_FP, n_features=s.NUM_FEATURES, memmapped=False)
# y = np.clip(y, 0, 1)
#
# batch_size = 1000
# num_classes = len(np.unique(y))
# epochs = 5
#
# print('Building model...')
# model = Sequential()
# model.add(Dense(512, input_shape=(X.shape[1],)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
#
# from tensorflow.python.client import device_lib
# print device_lib.list_local_devices()
#
# def roc_auc_score_keras(y_true, y_pred):
#     return roc_auc_score(y_true, y_pred)
#
# def mean_pred(y_true, y_pred):
#     someshit = K.mean(y_pred)
#     return someshit
#
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# # history = model.fit(X, y,
# #                     batch_size=batch_size,
# #                     epochs=epochs,
# #                     verbose=1,
# # validation_split=0.1)
#
# def batch_generator(X, y, batch_size):
#     number_of_batches = X.shape[0]/batch_size
#     counter=0
#     shuffle_index = np.arange(np.shape(y)[0])
#     np.random.shuffle(shuffle_index)
#     X =  X[shuffle_index, :]
#     y =  y[shuffle_index]
#     while 1:
#         index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
#         X_batch = X[index_batch,:].todense()
#         y_batch = y[index_batch]
#         counter += 1
#         yield(np.array(X_batch),y_batch)
#         if (counter < number_of_batches):
#             np.random.shuffle(shuffle_index)
#             counter=0
#
# model.fit_generator(generator=batch_generator(X, y, batch_size),
#                     epochs=epochs,
#                     steps_per_epoch=X.shape[0])
