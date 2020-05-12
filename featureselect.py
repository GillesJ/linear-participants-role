import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    Normalizer,
)
import datahandler as dh
import settings as s
from sklearn.datasets import dump_svmlight_file
from sklearn.utils.extmath import safe_sparse_dot
import time
from sklearn.feature_selection import SelectPercentile, f_classif
import logging
from scipy.special import ndtri
from cbrole_logging import setup_logging

setup_logging()
from sklearn.utils import check_X_y


def ig(X, y):
    def get_t1(fc, c, f):
        t = np.log2(fc / (c * f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fc, t)

    def get_t2(fc, c, f):
        t = np.log2((1 - f - c + fc) / ((1 - c) * (1 - f)))
        t[~np.isfinite(t)] = 0
        return np.multiply((1 - f - c + fc), t)

    def get_t3(c, f, class_count, observed, total):
        nfc = (class_count - observed) / total
        t = np.log2(nfc / (c * (1 - f)))
        t[~np.isfinite(t)] = 0
        return np.multiply(nfc, t)

    def get_t4(c, f, feature_count, observed, total):
        fnc = (feature_count - observed) / total
        t = np.log2(fnc / ((1 - c) * f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fnc, t)

    X = check_array(X, accept_sparse="csr")
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    # counts

    observed = safe_sparse_dot(Y.T, X)  # n_classes * n_features
    total = observed.sum(axis=0).reshape(1, -1).sum()
    feature_count = X.sum(axis=0).reshape(1, -1)
    class_count = (X.sum(axis=1).reshape(1, -1) * Y).T

    # probs

    f = feature_count / feature_count.sum()
    c = class_count / float(class_count.sum())
    fc = observed / total

    # the feature score is averaged over classes
    scores = (
        get_t1(fc, c, f)
        + get_t2(fc, c, f)
        + get_t3(c, f, class_count, observed, total)
        + get_t4(c, f, feature_count, observed, total)
    ).mean(axis=0)

    scores = np.asarray(scores).reshape(-1)

    return scores, []


def normal_CDF_inverse(p):
    """
    :param p: the probability to be inverted
    :return: the inverse of the cumulative distribution function of the Gaussian for the value p

    Following George Forman. (2007) "BNS Feature Scaling: An Improved Representation over TFIDF for SVM Text
    Classification". : "Finally, in the BNS function, the inverse Normal goes to infinity at zero or one; hence,
    we limit tpr and fpr to the range [0.0005, 1-0.0005]. Laplace smoothing is a more common method to avoid these
    extreme probabilities, but it damages the maximum likelihood estimate, and it loses the good performance of
    BNS by devaluing many valuable negative features in favor of very rare positive features [4]. Alternately and
    perhaps preferably, one could substitute a fractional count if tp or fp is exactly zero; this may work better
    for extremely large training sets. We used a fixed limit because we used a finite size lookup table for the
    inverse Normal function, [...]".
    """
    v = 0.00000001  # we constrain with a smaller v than Forman (cf comment above)
    p = max(v, min(p, 1 - v))

    assert p > 0.0 and p < 1

    # This is the scipy implementation of normal ppf inverse
    ncdfi = ndtri(p)
    return ncdfi


def bns(X, y, pos_label=1):

    X, y = check_X_y(X, y, ["csr", "csc", "coo"])

    pos_idc = np.flatnonzero(y == pos_label)
    neg_idc = np.flatnonzero(y != pos_label)
    positive_instances = X[pos_idc, :]
    negative_instances = X[neg_idc, :]
    pos, neg = len(pos_idc), len(neg_idc)

    tp = positive_instances.getnnz(axis=0).astype(
        "float32"
    )  # number of positive cases with the feature
    fp = negative_instances.getnnz(axis=0).astype(
        "float32"
    )  # number of negative cases with the feature

    tpr = np.divide(tp, pos)  # true positive rate
    fpr = np.divide(fp, neg)  # true negative rate

    normal_CDF_inverse_vectorized = np.vectorize(normal_CDF_inverse, otypes="f")
    bns_score = np.abs(
        normal_CDF_inverse_vectorized(tpr) - normal_CDF_inverse_vectorized(fpr)
    )  # BNS = |Finv(tpr) - Finv(fpr)|

    return bns_score


def main():

    X, y = dh.load_data(s.DATA_FP, n_features=s.NUM_FEATURES, memmapped=False)
    start = time.time()
    Xbns = SelectPercentile(bns, percentile=0.33).fit_transform(X, y)
    print(Xbns.shape)
    logging.info("Done BNS {} s".format(time.time() - start))
    start = time.time()
    Xf = SelectPercentile(f_classif, percentile=0.33).fit_transform(X, y)
    print(Xf.shape)
    logging.info("Done Anova F {} s".format(time.time() - start))


if __name__ == "__main__":
    main()
