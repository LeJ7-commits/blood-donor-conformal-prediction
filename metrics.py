import numpy as np
import pandas as pd


def empirical_coverage(y_true, pred_sets):
    """
    compute empirical coverage

    Returns
    coverage : float
    """
    y_true = np.asarray(y_true)
    covered = [(y_true[i] in pred_sets[i]) for i in range(len(y_true))]
    return float(np.mean(covered))


def average_set_size(pred_sets):
    """
    average size of prediction sets
    """
    return float(np.mean([len(s) for s in pred_sets]))


def coverage_by_group(y_true, pred_sets, groups):
    """
    compute coverage per subgroup (Mondrian evaluation)

    returns
    pd.DataFrame with columns:
        group, n, coverage, avg_set_size
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "pred_set": pred_sets,
        "group": groups
    })

    df["covered"] = df.apply(
        lambda r: r["y_true"] in r["pred_set"], axis=1
    )
    df["set_size"] = df["pred_set"].apply(len)

    return (
        df.groupby("group")
        .agg(
            n=("covered", "size"),
            coverage=("covered", "mean"),
            avg_set_size=("set_size", "mean")
        )
        .reset_index()
    )