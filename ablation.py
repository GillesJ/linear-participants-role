import util
from crossvalidate import select_model
import datahandler as dh
import settings as s
from cbrole_logging import setup_logging
import logging

setup_logging()


def add_types_to_groups(groups, types):
    full_groups = {}
    for groupname, typesubstrs in groups.iteritems():
        full_groups[groupname] = []
        full_groups[groupname].extend(
            [
                type
                for type in types
                for typesubstr in typesubstrs
                if typesubstr in type[0].fn
            ]
        )
    assert validate_feature_groups(full_groups, types)
    return full_groups


def get_consecutive_indices(g, validation=True):
    group_ind = {}
    for k, v in g.iteritems():
        range_tuples = [el[1] for el in v]
        range_corrected = [(el[0] - 1, el[1] - 1) for el in range_tuples]
        indices = []
        indices.extend([range(el[0], el[1]) for el in range_corrected])
        indices = flatten(indices)
        consecutive = []
        for key, group in groupby(
            enumerate(indices), lambda (index, item): index - item
        ):
            group = map(itemgetter(1), group)
            if len(group) > 1:
                consecutive.append((group[0], group[-1] + 1))
            else:
                consecutive.append((group[0], group[0] + 1))

        group_ind[k] = consecutive
    if validation:
        assert validate_group_ind(group_ind)
    return group_ind


def validate_group_ind(g):
    all_idc = []
    all_v = []
    for k, v in g.iteritems():
        all_v.extend(v)
    for el in all_v:
        all_idc.extend(range(el[0], el[1]))
    all_idc.sort()
    control = range(0, s.NUM_FEATURES)
    if control == all_idc:
        return True
    else:
        print list(set(control) - set(all_idc))
        return False


def validate_feature_groups(groups, types):
    # check if all types are in groups
    all_ingroup_values = []
    all_ingroup_values.extend([v for v in groups.itervalues()])
    all_ingroup_values = flatten(all_ingroup_values)
    ingroup_counts = len(set(all_ingroup_values))
    # check for doubles
    w = collections.defaultdict(list)
    for k, v in groups.iteritems():
        for i in v:
            w[i].append(k)
    doubles = [l for l in w.itervalues() if len(l) > 1]
    # print missing
    if len(doubles) != 0 or ingroup_counts != len(types):
        logging.warning(
            "Your groups are not properly defined. Doubles in groups: %s, Groups and mapdict type count "
            "difference: %s" % (doubles, ingroup_counts - len(types))
        )
        return False
    else:
        return True


def slice_feature_group(X, indices, name, dump=False):
    subgroup_data = []
    for rang in indices:
        start = rang[0]
        end = rang[1]
        X_subgroup = X[:, start:end]
        subgroup_data.append(X_subgroup)
    # make the full feature group by horizontal combination of subgrpup columns
    X_group = hstack(subgroup_data, format="csr")
    logging.debug(
        "{} made from original dataset indices {} (shape: {})".format(
            name, indices, X_group.shape
        )
    )
    # validate the amount of feature
    correct = flatten([range(rang[0], rang[1]) for rang in indices])
    assert X_group.shape[1] == len(correct)

    if dump:
        fg_fp = os.path.join(opt_dir, "%s.libsvm" % name)
        dump_svmlight_file(X_group, y, fg_fp, zero_based=False)
    return X_group


def unique_combine(l):
    combos = []
    for R in range(1, len(l) + 1):
        for subset in combinations(l, R):
            combos.append(subset)
    return sorted(combos)


def get_feature_group_indices(mapdict):
    fg_ranges = mapdict.get_gallop_feature_ranges()
    groups = add_types_to_groups(s.FEATURE_GROUPS, fg_ranges)
    indices = get_consecutive_indices(groups)
    return indices


def combine_feature_groups(fgs):
    """
    :param fgs: dict of feature group indices.
    :return: all possible combinations
    """
    keys = [k for k in fgs.iterkeys()]
    vals = [v for v in fgs.itervalues()]
    keycombo = unique_combine(keys)
    valscombo = unique_combine(vals)
    keycombo = ["+".join(sorted(list(k))) for k in keycombo]
    valscombo = [sorted(flatten(list(combo))) for combo in valscombo]
    combo_indices = dict(zip(keycombo, valscombo))
    # make consecutive
    consecutive_combo_idc = {}
    for k, v in combo_indices.iteritems():
        indices = []
        indices.extend([range(el[0], el[1]) for el in v])
        indices = flatten(indices)
        consecutive = []
        for key, group in groupby(
            enumerate(indices), lambda (index, item): index - item
        ):
            group = map(itemgetter(1), group)
            if len(group) > 1:
                consecutive.append((group[0], group[-1] + 1))
            else:
                consecutive.append((group[0], group[0] + 1))
        consecutive_combo_idc[k] = consecutive

    return consecutive_combo_idc


def main():
    global dataset_name
    # load data
    X, y = dh.load_data(s.DATA_FP, n_features=s.NUM_FEATURES, memmapped=False)
    X, y = X[0:10000, :], y[0:10000]

    # Make feature group combinations for Devset and holdout
    mapdict = dh.load_mapdict_file(s.MAPDICT_FP)
    fg_indices = get_feature_group_indices(mapdict)
    combo_indices = combine_feature_groups(fg_indices)

    # fit the ablation experiment
    ablation_report = {}
    for ablation_name, indices in combo_indices.iteritems():
        dataset_name = ablation_name
        X_dev_abl = slice_feature_group(X_dev, indices, ablation_name)
        X_holdout_abl = slice_feature_group(X_holdout, indices, ablation_name)

        logging.info(
            "\n=============================================================="
            "\n{}: Performing crossvalidation."
            "\n==============================================================".format(
                ablation_name.upper()
            )
        )

        winner_report = crossvalidate(
            X_dev_abl, y_dev, X_holdout_abl, y_holdout, clf, cv
        )
        ablation_report["true_labels_holdout"] = y_holdout.tolist()
        ablation_report[ablation_name] = winner_report

    with open(
        os.path.join(s.OPT_DIRP, "%s_ablation_report.json" % timestamp), "wt"
    ) as f:
        json.dump(ablation_report, f, sort_keys=True)

    pprint(ablation_report, depth=20)
    select_model(X, y)


if __name__ == "__main__":
    main()
