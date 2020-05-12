"""
script to run a cross validation experiment
"""
import pipeline
import settings as s
import datahandler as dh
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)
import cPickle as pickle
import util
from sklearn.externals.joblib import dump
from svmlight_loader import load_svmlight_file, dump_svmlight_file
import timeit
import json
import datetime
import os
from cbrole_logging import setup_logging
import logging

setup_logging()


def select_model(X, y):
    # make holdout-holdin split
    (
        X_in,
        X_out,
        y_in,
        y_out,
        indices_in,
        indices_out,
        removed_features,
    ) = dh.split_holdout(X, y)

    logging.info("Writing holdin-holdout split data and info to file.")

    dump_svmlight_file(
        X_in, y_in, os.path.join(s.OPT_DIRP, "holdin.svm"), zero_based=True
    )
    dump_svmlight_file(
        X_out, y_out, os.path.join(s.OPT_DIRP, "holdout.svm"), zero_based=True
    )
    with open(os.path.join(s.OPT_DIRP, "holdinout_split_indices.json"), "wt") as f:
        json.dump(
            {
                "holdin": indices_in.tolist(),
                "holdout": indices_out.tolist(),
                "num_features": X_in.shape[1],
            },
            f,
        )

    steps, param_grids = pipeline.make_pipelines(s.PIPE_STEPS, alt_order=s.ALT_ORDER)
    steps_param_grids = zip(steps, param_grids)

    if (
        s.PARTIALRUN
    ):  # filter with partial run info from the list pkl generated by reporter.py
        partialinfo = json.load(open(s.PARTIALRUN, "rt"))
        steps_param_grids = pipeline.filter_partialrun(steps_param_grids, partialinfo)

    all_results = {}
    fit_pred_duration = {}
    cv_pipe_dir = os.path.join(s.OPT_DIRP, "cv_pipelines")
    util.ensure_dir(cv_pipe_dir)

    for (steps, param_grid) in steps_param_grids:

        # generate a human readable name for the current pipeline from the Pipeline object
        pipe_name = []
        for (name, step) in steps:
            if not "SelectPercentile" in str(step):
                pipe_name.append(str(step).split("(")[0].lower())
            else:
                pipe_name.append(str(step.score_func.func_name).split("(")[0].lower())
        pipe_name = "+".join(pipe_name)
        DATASET_NAME = "{}_{}".format(
            pipe_name, s.DATASET_NAME
        )  # append the dataset name with pipeline name for
        # logging and metadata purposes
        pipe_opt_dir = os.path.join(cv_pipe_dir, pipe_name)
        util.ensure_dir(pipe_opt_dir)

        pipe = Pipeline(steps)
        grid_search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=pipeline.my_scorer,
            n_jobs=s.CV_N_JOBS,
            cv=s.CV,
            verbose=10,
            error_score=0,
            return_train_score=False,
        )

        logging.info("{}: Doing modelselection with {}.".format(pipe_name, grid_search))
        start_pipefit = timeit.default_timer()
        grid_search.fit(X_in, y_in)

        # save grid_search object
        logging.info("{}: Pickling crossvalidation object..".format(pipe_name))
        dump(
            grid_search,
            os.path.join(pipe_opt_dir, "%s_grid_search.joblibpkl" % s.TIMESTAMP),
            compress=1,
        )

        # save all intermediate results
        all_results[pipe_name] = grid_search.cv_results_
        with open(
            os.path.join(s.OPT_DIRP, "all_pipeline_cv_results.pkl"), "wb"
        ) as all_res_out:
            pickle.dump(all_results, all_res_out)

        logging.info(
            "{}: Evaluating winning model on holdout test set.".format(pipe_name)
        )

        logging.info("{}: Evaluating holdout performance.".format(pipe_name))
        y_pred = grid_search.predict(X_out).astype(int)
        y_out_true_y_out_pred = {
            "y_out_true": y_out.tolist(),
            "y_out_pred": y_pred.tolist(),
        }
        with open(os.path.join(pipe_opt_dir, "y_out_true-y_out_pred.json"), "wt") as f:
            json.dump(y_out_true_y_out_pred, f)

        # save all intermediate fit and predict durations
        elapsed = timeit.default_timer() - start_pipefit
        fit_pred_duration[pipe_name] = elapsed
        json.dump(
            fit_pred_duration,
            open(
                os.path.join(s.OPT_DIRP, "all_pipeline_fit_predict_duration.json"), "wt"
            ),
        )

        precision, recall, fscore, support = precision_recall_fscore_support(
            y_out, y_pred, average=s.SCORE_AVERAGING
        )
        acc = accuracy_score(y_out, y_pred)
        if s.MULTICLASS:
            auc = None
        else:
            auc = roc_auc_score(y_out, y_pred)

        # make report
        params = grid_search.best_params_
        winscore = grid_search.best_score_
        ablation_name = "blah"
        report = (
            "%s\t%s\t%s"
            "\nSettings: %s"
            "\nTested parameters: %s"
            "\nWinning parameters: %s"
            "\nWinning model CV score: %s %s"
            "\nHoldout score:"
            "\nfscore\tprecision\trecall\tacc\tauc"
            "\n%s\t%s\t%s\t%s\t%s"
            % (
                s.DATA_FP,
                ablation_name,
                str(pipe.get_params()),
                s.__file__,
                s.PIPE_STEPS,
                params,
                winscore,
                s.SCORER_METRIC,
                fscore,
                precision,
                recall,
                acc,
                auc,
            )
        )
        print(report)
        with open(
            os.path.join(pipe_opt_dir, "%s_results.txt" % s.TIMESTAMP), "wt"
        ) as f:
            f.write(report)
        report_as_dict = {
            "data_path": s.DATA_FP,
            "feature_groups": ablation_name,
            # 'classifier_type': str(type(clf)),
            "settings": str(s.__file__),
            "param_grid": str(s.PIPE_STEPS),
            "best_params": str(params),
            "score_grid_search": winscore,
            "metric_grid_search": s.SCORER_METRIC,
            "fscore_holdout": fscore,
            "precision_holdout": precision,
            "recall_holdout": recall,
            "acc_holdout": acc,
            "auc_holdout": auc,
            "support_holdout": support,
            "predictions_holdout": y_pred.tolist(),
            "y_true_holdout": y_out.tolist(),
        }

        with open(
            os.path.join(pipe_opt_dir, "%s_finalreport.txt" % s.TIMESTAMP), "wt"
        ) as f:
            f.write(report)
        with open(os.path.join(pipe_opt_dir, "report.json"), "wt") as f:
            json.dump(report_as_dict, f)

        logging.info(
            "{}: Model selection done. Duration: {}".format(
                pipe_name.upper(), str(datetime.timedelta(seconds=elapsed))
            )
        )

    logging.info("DONE.")


def main():
    # load data
    X, y = dh.load_data(s.DATA_FP, n_features=s.NUM_FEATURES, memmapped=False)

    # # # TESTING
    # print("Warning TESTING")
    # X, y = X[0:1000, :], y[0:1000]
    # logging.warning("TESTING with {}".format(X.shape))

    util.ensure_dir(s.OPT_DIRP)
    X = dh.do_memmap(X)

    select_model(X, y)
    util.send_text_message(
        "{}: Ended run and wrote all to {}".format(
            str(datetime.datetime.now()), s.OPT_DIRP
        )
    )


if __name__ == "__main__":
    main()
