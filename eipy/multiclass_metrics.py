import numpy as np
import pandas as pd
import inspect
from eipy.utils import minority_class, predictive_multiclass_data
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score
from tqdm import tqdm


def fmax_score(y_test, y_score, beta=1.0, pos_label=1):
    fmax_score, _, _, threshold_fmax = fmax_precision_recall_threshold(
        y_test, y_score, beta=beta, pos_label=pos_label
    )
    return fmax_score, threshold_fmax


def fmax_precision_recall_threshold(labels, y_score, beta=1.0, pos_label=1):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein
    Function Prediction. Nature Methods, 10(3), 221-227.
    Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In
    Introduction to Information Retrieval. Cambridge University Press.
    """
    if pos_label == 0:
        labels = 1 - np.array(labels)
        y_score = 1 - np.array(y_score)

    precision_scores, recall_scores, thresholds = precision_recall_curve(
        labels, y_score
    )

    np.seterr(divide="ignore", invalid="ignore")
    f_scores = (
        (1 + beta**2)
        * (precision_scores * recall_scores)
        / ((beta**2 * precision_scores) + recall_scores)
    )

    arg_fmax = np.nanargmax(f_scores)

    fmax_score = f_scores[arg_fmax]
    precision_fmax = precision_scores[arg_fmax]
    recall_fmax = recall_scores[arg_fmax]
    threshold_fmax = thresholds[arg_fmax]

    return fmax_score, precision_fmax, recall_fmax, threshold_fmax

#make this automatically detect num classes
def nested_threshold_fmax(y_test, y_pred):
  fmax = [0,0,0]
  thresholds = [0,0,0]

  all_indices = np.arange(y_pred.shape[0])

  thresh_space_0 = y_pred[:,0]
  for thresh_0 in thresh_space_0:
    cn_indices = np.where(y_pred[:,0]> thresh_0)[0]
    ci_indices = np.setdiff1d(all_indices, cn_indices)

    if len(ci_indices) != 0:
      ci_preds = [(idx, y_pred[idx,1:]) for idx in ci_indices]
      ci_preds_norm = [(v[0], v[1]/np.sum(v[1])) for v in ci_preds]

      thresh_1_space = [v[1][0] for v in ci_preds_norm]
      for thresh_1 in thresh_1_space:
        y_decision = np.zeros(y_pred.shape[0])
        mci_indices = [v[0] for v in ci_preds_norm if v[1][0]>thresh_1]
        ad_indices = np.setdiff1d(ci_indices,mci_indices)

        y_decision[ci_indices] = 1
        y_decision[ad_indices] = 2
        f1s = f1_score(y_test, y_decision, average=None)
        for i in range(3):
          if f1s[i] >= fmax[i]:
            fmax[i] = f1s[i]
            thresholds[i] = (thresh_0,thresh_1)
    else:
      y_decision = np.zeros(y_pred.shape[0])
      f1s = f1_score(y_test, y_decision, average=None)

      for i in range(3):
        if f1s[i] >= fmax[i]:
          fmax[i] = f1s[i]
          thresholds[i] = (thresh_0,np.nan)
  #rearrange for plotting  
  f_t = [(fmax[i], thresholds[i]) for i in range(3)]

  return f_t

#generalize for using n_clsases
def decide_with_thresholds(y_pred, t_pair):
    all_indices = np.arange(y_pred.shape[0])
    
    y_decision = np.zeros(y_pred.shape[0])
    cn_indices = np.where(y_pred[:,0]> t_pair[0])
    ci_indices = np.setdiff1d(all_indices, cn_indices)

    ci_preds = [(idx, y_pred[idx,1:]) for idx in ci_indices]
    ci_preds_norm = [(v[0], v[1]/np.sum(v[1])) for v in ci_preds]

    mci_indices = [v[0] for v in ci_preds_norm if v[1][0]>t_pair[1]]
    ad_indices = np.setdiff1d(ci_indices,mci_indices)

    y_decision[ci_indices] = 1
    y_decision[ad_indices] = 2
    
    return y_decision

#alt refers to how the threshold space is generated. Fixed vs observed.
def alt_nested_threshold_fmax(y_true, y_pred, target_class=-1):
    n_classes = len(np.unique(y_true))
    fmax = np.zeros(shape=(n_classes))
    thresholds_max = None

    thresholds = np.arange(0,1,.01)
    grid_1, grid_2 = np.meshgrid(thresholds, thresholds)
    threshold_pairs = np.vstack([grid_1.flatten(), grid_2.flatten()]).T
    for t_pair in tqdm(threshold_pairs, desc='Computing class specfic fmax scores for base model'):
        y_decision = decide_with_thresholds(y_pred, t_pair=t_pair)
        f1s = f1_score(y_true, y_decision, average=None)
        if f1s[target_class] > fmax[target_class]:
            fmax = f1s
            thresholds_max = t_pair

    return fmax, thresholds_max


def try_metric_with_pos_label(y_true, y_pred, metric, pos_label):
    """
    Compute score for a given metric.
    """
    if "pos_label" in inspect.signature(metric).parameters:
        score = metric(y_true, y_pred, pos_label=pos_label)
    else:
        score = metric(y_true, y_pred)
    return score


def scores(y_true, y_pred):
    """
    Compute all metrics for a single set of predictions. Returns a dictionary
    containing metric keys, each paired to a tuple (score, threshold).
    """
    
    y_pred = np.array(y_pred.to_list())
    y_pred = np.argmax(y_pred, axis=-1)
    f1s = f1_score(np.array(y_true),y_pred, average=None)

    metric_threshold_dict = {f"f1 class {i}": (f1s[i], "argmax") for i in range(len(f1s))}
    metric_threshold_dict['macro avg f1'] = (f1_score(np.array(y_true),y_pred, average='macro'), 'argmax')
        # if isinstance(y_pred[0], np.ndarray):
        #     y_pred = [np.argmax(y) for y in  y_pred]
        # precision_macro = precision_score(y_true, y_pred, average='macro')
        # recall_macro = recall_score(y_true, y_pred, average='macro')
        # f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # precision_micro = precision_score(y_true, y_pred, average='micro')
        # recall_micro = recall_score(y_true, y_pred, average='micro')
        # f1_micro = f1_score(y_true, y_pred, average='micro')

        # metric_threshold_dict = {
        #     "precision (macro)" : (precision_macro, "argmax"),
        #     "recall (macro)" : (recall_macro, "argmax"),
        #     "f1 (macro)" : (f1_macro,"argmax"),
        #     "precision (micro)" : (precision_micro, "argmax"),
        #     "recall (micro)" : (recall_micro, "argmax"),
        #     "f1 (micro)" : (f1_micro,"argmax"),
        # }

    return metric_threshold_dict


def scores_matrix(X, labels):
    """
    Calculate metrics and threshold (if applicable) for each column
    (set of predictions) in matrix X
    """

    scores_dict = {}
    for column in X.columns:
        column_temp = X[column]
        metrics_per_column = scores(labels, column_temp)
        # metric_names = list(metrics.keys())
        for metric_key in metrics_per_column.keys():
            if not (metric_key in scores_dict):
                scores_dict[metric_key] = [metrics_per_column[metric_key]]
            else:
                scores_dict[metric_key].append(metrics_per_column[metric_key])

    return scores_dict


def create_metric_threshold_dataframes(X, labels):
    """
    Create a separate dataframe for metrics and thresholds. thresholds_df contains
    NaN if threshold not applicable.
    """

    scores_dict = scores_matrix(X, labels)

    metrics_df = pd.DataFrame(columns=X.columns)
    thresholds_df = pd.DataFrame(columns=X.columns)
    for k, val in scores_dict.items():
        metrics_df.loc[k], thresholds_df.loc[k] = list(zip(*val))
    return metrics_df, thresholds_df


def create_metric_threshold_dict(X, labels):
    df_dict = {}
    df_dict["metrics"], df_dict["thresholds"] = create_metric_threshold_dataframes(
        X, labels
    )
    return df_dict


def mc_base_summary(ensemble_test_dataframes):
    """
    Create a base predictor performance summary by concatenating data across test folds
    """

    labels = pd.concat([df["labels"] for df in ensemble_test_dataframes])

    #for multiple samples
    ensemble_test_averaged_samples = pd.concat(
        [
            df.drop(columns=["labels"], level=0).groupby(level=(0, 1, 3), axis=1).mean()
            for df in ensemble_test_dataframes
        ]
    )
    #df of probability vectors for every modality/bp pair for every sample in data
    ensemble_test_aggregated_predictions = ensemble_test_averaged_samples.groupby(level=(0,1), axis=1).agg(list)
    return create_metric_threshold_dict(ensemble_test_aggregated_predictions, labels)


def mc_ensemble_summary(ensemble_predictions):
    X = ensemble_predictions.drop(["labels"], axis=1)
    labels = ensemble_predictions["labels"]
    X = X.groupby(level=(0), axis=1).agg(lambda x: x.values.tolist())
    return create_metric_threshold_dict(X, labels)


# These two functions are an attempt at maximizing/minimizing any metric
# def metric_scaler_function(arg, y_true, y_pred, metric, pos_label, multiplier):
#         threshold = np.sort(np.unique(y_pred))[int(np.round(arg))]
#         y_binary = (y_pred >= threshold).astype(int)
#         return multiplier * try_metric_with_pos_label(y_true, y_binary, metric, pos_label)

# def max_min_score(y_true, y_pred, metric, pos_label, max_min):
#     '''
#     Compute maximized/minimized score for a given metric.
#     '''

#     if max_min=='max':
#         multiplier = -1
#     elif max_min=='min':
#         multiplier = 1

#     optimized_result = minimize_scalar(
#                                         metric_scaler_function,
#                                         args=(y_true, y_pred, metric, pos_label, multiplier),
#                                         bounds=(0, len(np.unique(y_pred))-1),
#                                         method='bounded'
#                                         )

#     threshold = np.sort(np.unique(y_pred))[int(np.round(optimized_result.x))]
#     score = multiplier * optimized_result.fun

#     return score, threshold
#
