"""
Classes and methods for handling and making an sklearn-compatible pipeline.

"""
import copy
import itertools
import json
import logging
import multiprocessing
from time import time

import os
from sklearn import metrics
from sklearn.externals.joblib import dump
from sklearn.model_selection import ParameterGrid
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.pipeline import Pipeline
import more_itertools
import settings as s
from collections import OrderedDict
import numpy as np
import util
from cbrole_logging import setup_logging

setup_logging()


def check_estimator_grid(est, grid):

    estkeys, gridkeys = dict(est).keys(), dict(grid).keys()
    grid_steps = set([k.split("__")[0] for k in gridkeys])
    est_steps = set(estkeys)
    diff1 = grid_steps.difference(est_steps)
    diff2 = est_steps.difference(grid_steps)
    if diff1 == diff2 == set():
        pass
    else:
        raise ValueError(
            "Discrepancy between pipeline estimator steps and param grid for entry {} {}.".format(
                list(diff1), list(diff2)
            )
        )


def make_paramgrid(estimator_steps, grid):
    def check_emb_params(params):
        checked = {k: [v] for k, v in params.iteritems() if not isinstance(v, list)}
        params.update(checked)
        return params

    def parametrize_estimator(estim, params):

        all_parametrized_estims = []
        param_grid = ParameterGrid(params)
        for prms in param_grid:
            parametrized_estim = copy.copy(estim)
            parametrized_estim.set_params(**prms)
            all_parametrized_estims.append(parametrized_estim)

        return all_parametrized_estims

    check_estimator_grid(estimator_steps, grid)

    sk_param_grid = {}

    for (step, estimparams) in grid:
        sk_param_grid[step] = []
        for (estim, params) in estimparams:
            # deal with None
            if estim == None:
                sk_param_grid[step].append(None)

            # deal with embedded Pipeline
            elif isinstance(estim, Pipeline):
                emb_pipeline_steps = estim.steps
                assert len(emb_pipeline_steps) == len(
                    params
                )  # the dict def must specify a param grid for each step. No params specified by empty dict.
                if not isinstance(params, list):
                    list(params)

                emb_parametrized_steps = {}
                for (emb_step, emb_params) in zip(emb_pipeline_steps, params):
                    emb_params = check_emb_params(emb_params)
                    emb_step_name = emb_step[0]
                    emb_step_estim = emb_step[1]
                    all_parametrized_emb_estims = []
                    emb_param_grid = ParameterGrid(emb_params)
                    for emb_prms in emb_param_grid:
                        emb_parametrized_estim = copy.copy(emb_step_estim)
                        emb_parametrized_estim.set_params(**emb_prms)
                        all_parametrized_emb_estims.append(emb_parametrized_estim)

                    emb_parametrized_steps[emb_step_name] = all_parametrized_emb_estims

                # MAKE ORDEREDDICT OF WITH KEYS STEPNUMBERS
                steps_in_order = [
                    [(step_n, el) for el in emb_parametrized_steps.get(step_n)]
                    for step_n in [j[0] for j in estim.steps]
                ]
                steps_combo = list(itertools.product(*steps_in_order))
                all_pipelines_emb = [Pipeline(pipestep) for pipestep in steps_combo]
                sk_param_grid[step] = all_pipelines_emb

            # return a list of estimator objects with al possible parameters
            else:
                parametrized_estims = parametrize_estimator(estim, params)
                sk_param_grid[step].extend(parametrized_estims)

    return sk_param_grid


def log_metrics_and_params(
    clf, results, fold_log_dir, y_true=[], y_pred=[], model_save_fp=None, log_name=None
):
    # log results and save path
    clf_type = type(clf)
    clf_params = clf.get_params()
    logging.info("\nResults:\t{}\nPipeline:\t{}\n".format(results, clf_params))
    to_write = {}
    to_write["pipe_name"] = log_name
    to_write["results"] = results
    to_write["y_pred"] = y_pred.tolist()
    to_write["y_true"] = y_true.tolist()
    to_write["clf_type"] = str(clf_type)
    to_write["clf_params"] = str(clf_params)
    to_write["savepath_model"] = model_save_fp
    # pprint(to_write)
    try:
        current_proc = multiprocessing.current_process()
        proc_str = "{}{}{}".format(
            current_proc.name, current_proc._identity, current_proc.pid
        )
        intermed_result_proc_fp = os.path.join(
            fold_log_dir, "{}_{}.json".format(s.TIMESTAMP, proc_str)
        )
        with open(intermed_result_proc_fp, mode="a") as int_f:
            json.dump(to_write, int_f, sort_keys=True)
            int_f.write("{}".format(os.linesep))
    except Exception as e:
        logging.exception("Could not write intermediate result.")


def save_model(clf, fold_model_dir, log_name=None):
    # save model with timestamp
    timestring = str(time()).replace(".", "")

    if log_name:
        savepath_suffix = "{}_{}".format(timestring, log_name)
    else:
        savepath_suffix = "{}".format(timestring)

    model_savepath = os.path.join(
        fold_model_dir, "model_{}.pkl".format(savepath_suffix)
    )
    try:
        dump(clf, model_savepath, compress=1)
    except Exception as e:
        logging.exception("Failed to pickle candidate classifier.")

    return model_savepath


def get_metrics(y_true=[], y_pred=[]):
    # compute more than just one metrics

    chosen_metrics = {
        "f1": metrics.f1_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "accuracy": metrics.accuracy_score,
        "auc": metrics.roc_auc_score,
    }
    results = {}
    for metric_name, metric_func in chosen_metrics.items():
        try:
            if metric_name == "auc" and s.MULTICLASS:
                inter_res = None
            elif metric_name in ["f1", "precision", "recall"]:
                inter_res = metric_func(y_true, y_pred, average=s.SCORE_AVERAGING)
            else:
                inter_res = metric_func(y_true, y_pred)
        except Exception as ex:
            inter_res = None
            logging.exception("Couldn't evaluate %s because of %s", metric_name, ex)
        results[metric_name] = inter_res

    return results


def extract_clf_name(clf):
    if isinstance(clf, imbPipeline):
        pipe_name = []
        for (name, step) in clf.steps:
            if not "SelectPercentile" in str(step):
                method = str(step).split("(")[0].lower()
            else:
                method = "" + str(step.score_func.func_name).split("(")[0].lower()
            stepnamed = "{}{}".format(name.lower(), method.upper())
            pipe_name.append(stepnamed)
        pipe_name = "+".join(pipe_name)

    return pipe_name


def my_scorer(clf, X_val, y_true_val):

    log_name = extract_clf_name(clf)
    metric = s.SCORER_METRIC
    fold_log_dirp = s.SCORER_FOLD_LOG_DIRP
    util.ensure_dir(fold_log_dirp)
    fold_model_dirp = s.SCORER_FOLD_MODEL_DIRP
    util.ensure_dir(fold_model_dirp)

    # do all the work and return some of the metrics
    y_pred_val = clf.predict(X_val)
    results = get_metrics(y_true=y_true_val, y_pred=y_pred_val)

    model_save_fp = None
    if fold_model_dirp:
        model_save_fp = save_model(clf, fold_model_dirp, log_name=log_name)

    if fold_log_dirp:
        log_metrics_and_params(
            clf,
            results,
            fold_log_dirp,
            y_true=y_true_val,
            y_pred=y_pred_val,
            model_save_fp=model_save_fp,
            log_name=log_name,
        )

    return results[metric]


def make_estim_step(param_grid):
    """

    :param param_grids: list of all param_grid, one param_grid for one pipeline
    :return:
    """
    return list(
        more_itertools.unique_everseen(
            [(k.split("__")[0], v) for k, v in param_grid.iteritems()],
            key=lambda x: x[0],
        )
    )


def format_param_grid(param_grid):

    formatted = OrderedDict()
    for k, v in param_grid.iteritems():
        if isinstance(v, list) or isinstance(v, np.ndarray):
            formatted[k] = v
    return formatted


def make_pipelines(steps, alt_order=None):
    step_names = [step[0] for step in steps]
    step_estims = [step[1] for step in steps]
    # make a combination of all estimators
    estim_combos = list(itertools.product(*step_estims))
    name_combos = [tuple(step_names) for i in range(len(estim_combos))]
    if alt_order:
        if not any(isinstance(i, list) for i in alt_order):
            alt_order = [alt_order]
        for alt_ord in alt_order:
            alt_step_estims = []
            for step_n in alt_ord:
                alt_step_estims.append(step_estims[step_names.index(step_n)])
        alt_estim_combos = list(itertools.product(*alt_step_estims))
        estim_combos.extend(alt_estim_combos)
        name_combos.extend([tuple(alt_ord) for i in range(len(alt_estim_combos))])

    dup_idc = [
        idx for idx, item in enumerate(estim_combos) if item in estim_combos[:idx]
    ]
    estim_combos = util.remove_indexes(estim_combos, dup_idc)
    name_combos = util.remove_indexes(name_combos, dup_idc)

    # make param_grids
    all_param_grids = []
    for step_names, estim_param_pair in zip(name_combos, estim_combos):
        # make the dict
        param_grid = OrderedDict()
        for name, estim_param in zip(step_names, estim_param_pair):

            if isinstance(estim_param, list):
                for i, (emb_estim, emb_param) in enumerate(estim_param):

                    if emb_estim:
                        emb_name = "{}{}".format(name, i)
                        param_grid[emb_name] = emb_estim

                        if emb_param:  # if param is empty ignore
                            for emb_param_name, emb_val in emb_param.iteritems():
                                emb_param_grid_name = "{}__{}".format(
                                    emb_name, emb_param_name
                                )
                                param_grid[emb_param_grid_name] = emb_val

            elif isinstance(estim_param, tuple):

                if (
                    estim_param[0] is not None
                ):  # do not add step if the estimator does not exist
                    param_grid[name] = estim_param[0]

                    if estim_param[1]:
                        for param_name, val in estim_param[1].iteritems():
                            param_grid_name = "{}__{}".format(name, param_name)
                            param_grid[param_grid_name] = val
            else:
                logging.warning(
                    "Formatting of steps in settings is wrong. Could not generate paramgrid for pipeline for {}{}.".format(
                        name, estim_param
                    )
                )

        if param_grid not in all_param_grids:
            all_param_grids.append(param_grid)

    step_init = [make_estim_step(p) for p in all_param_grids]
    param_grids = [format_param_grid(p) for p in all_param_grids]

    return step_init, param_grids


def filter_partialrun(steps_param_grids, partialinfo):

    filt_steps_param_grids = []
    for (steps, param_grid) in steps_param_grids:

        # generate a human readable name for the current pipeline from the Pipeline object
        pipe_name = []
        for (name, step) in steps:
            if not "SelectPercentile" in str(step):
                pipe_name.append(str(step).split("(")[0].lower())
            else:
                pipe_name.append(str(step.score_func.func_name).split("(")[0].lower())
        pipe_name = "+".join(pipe_name)

        if pipe_name not in partialinfo:
            filt_steps_param_grids.append((steps, param_grid))

    return filt_steps_param_grids
