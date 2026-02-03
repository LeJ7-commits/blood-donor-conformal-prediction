import numpy as np


def compute_nonconformity_scores(proba, y_true):
    """
    compute nonconformity scores: 1 - p_true

    Returns
    scores : np.ndarray
        Nonconformity scores
    """
    y_true = np.asarray(y_true)
    return 1.0 - proba[np.arange(len(y_true)), y_true]


def conformal_quantile(scores, alpha):
    """
    compute conformal quantile q_hat

    Uses:
        q_hat = ceil((n + 1) * (1 - alpha)) / n

    Returns
    qhat : float
    """
    scores = np.sort(np.asarray(scores))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return scores[k - 1]


def split_conformal_prediction_sets(proba_test, qhat):
    """
    generate prediction sets for split conformal classification

    Include class c if:
        p_c >= 1 - qhat

    Returns
    pred_sets : list of sets
        Prediction sets per sample
    """
    threshold = 1.0 - qhat
    pred_sets = []

    for row in proba_test:
        pred_sets.append(set(np.where(row >= threshold)[0]))

    return pred_sets


def mondrian_qhats(scores, groups, alpha):
    """
    compute group-conditional (Mondrian) q_hat values

    Returns
    qhats : dict
        Mapping group -> qhat
    """
    qhats = {}
    groups = np.asarray(groups)

    for g in np.unique(groups):
        group_scores = scores[groups == g]
        qhats[g] = conformal_quantile(group_scores, alpha)

    return qhats
