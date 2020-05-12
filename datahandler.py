"""
Handles data
"""
import atexit
import logging
import settings as s
import shutil
import tempfile
import os
import numpy as np
from cbrole_logging import setup_logging
from svmlight_loader import load_svmlight_file, dump_svmlight_file
from sklearn.externals.joblib import Memory, dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import json

setup_logging()


def do_memmap(X):
    try:
        logging.info("Memmapping instances to temp folder.")
        # saving dataset X to a local file for memmapping
        temp_folder = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, temp_folder)  # remove this at exit
        X_memmap_fp = os.path.join(temp_folder, "X.mmap")
        dump(X, X_memmap_fp)
        # loading for memmapped usage
        X = load(X_memmap_fp, mmap_mode="w+")
        logging.debug("Memmapped dataset instances to %s" % X_memmap_fp)
        return X

    except Exception as e:
        logging.exception("Memmaping the instances of dataset failed.")


def load_data(fp, n_features=None, memmapped=False, **kwargs):
    """Loads a file in given vector format to CSR as required by scikit-featureselection
    """
    mem = Memory(".mycache_%s" % os.path.basename(os.path.normpath(fp)), verbose=False)

    @mem.cache
    def get_data(in_filename, n_features, **kwargs):
        data = load_svmlight_file(in_filename, n_features=n_features, dtype=np.float32)
        return data[0], data[1]

    logging.info('Loading dataset "%s".' % fp)

    X, y = get_data(fp, n_features, **kwargs)

    if (
        memmapped
    ):  # this is not currently working because bug in sklearn 0.18 is milestone by 0.19
        X = do_memmap(X)

    y = y.astype(int)

    logging.info("Data: %d instances, %d features." % (X.shape[0], X.shape[1]))
    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    logging.info("Class distribution: %s." % class_counts)

    return X, y


def drop_cols(M, idx_to_drop):
    """Remove columns from matrix M given a list of column indices to remove."""
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()


def get_zero_cols(X):
    """Returns the indices of columns for which the sum of all values equals
    the threshold.
    """
    # get sum over columns
    sums = X.sum(axis=0).A.flatten()
    cols_to_drop = [i for i, sum in enumerate(sums) if sum == 0.0]
    # double check if indeed all zero: get the nonzero counts per column, assert all of those are zero
    nonzero_cnts = X[:, cols_to_drop].getnnz(axis=0)
    assert np.count_nonzero(nonzero_cnts) == 0
    # # manual check
    # for i in cols_to_drop[:10]:
    #     col = X[:, i].todense().flatten()[0]
    #     assert all(c == 0 for c in col)
    return cols_to_drop


def split_holdout(X, y):
    # make holdout - split
    no_valid_split = True
    while no_valid_split:

        logging.info("Making random split holdout of %s size." % s.HOLDOUT)
        indices = np.arange(X.shape[0])

        X_in, X_out, y_in, y_out, indices_in, indices_out = train_test_split(
            X, y, indices, test_size=s.HOLDOUT, random_state=s.RANDOM_STATE
        )

        dev_class_counts = np.asarray(np.unique(y_in, return_counts=True)).T.tolist()
        holdout_class_counts = np.asarray(
            np.unique(y_out, return_counts=True)
        ).T.tolist()
        logging.info(
            "Holdout class distribution: %s. Devset class distribution: %s."
            % (holdout_class_counts, dev_class_counts)
        )

        if (
            np.unique(y_in,).size != np.unique(y).size
            or np.unique(y_out).size != np.unique(y).size
        ):
            s.RANDOM_STATE += 1
            logging.warning(
                "Label missing in devset or holdout set. Remaking random heldout-heldin split with NEW \
                            RANDOMSTATE: {}.".format(
                    s.RANDOM_STATE
                )
            )
        else:
            no_valid_split = False

    # remove unseen features indices in hold-in from hold-out
    # unseen feature is a feature that is discrete (non-continuous, i.e. n-gram counts) with value 0
    # get all idc with value 0 in held-in
    X_in_zero_cols_idc = get_zero_cols(X_in)
    # check that these have a value other than 0 in heldout, if yes: remove, if not they are continuous variables such as topic similarity
    X_out_to_filter = X_out[:, X_in_zero_cols_idc]
    X_out_nnz = X_out_to_filter.getnnz(axis=0)
    cols_to_drop = [
        X_in_zero_cols_idc[i] for i, nnz in enumerate(X_out_nnz) if nnz != 0
    ]
    logging.info(
        "Removing {} feature columns for features that are not in hold-in set.".format(
            len(cols_to_drop)
        )
    )
    X_in = drop_cols(X_in, cols_to_drop)
    X_out = drop_cols(X_out, cols_to_drop)

    return X_in, X_out, y_in, y_out, indices_in, indices_out, cols_to_drop


def load_mapdict_file(mapdict_fp):
    try:
        logging.info("Loading mapdict from file (%s)." % mapdict_fp)
        with open(mapdict_fp, "rb") as f:
            mapdict = pickle.load(f)
        logging.debug("Succesfully loaded mapdict from file (%s)." % mapdict_fp)
        return mapdict
    except:
        logging.exception("Could not load mapdict.")
        return None
