import pytest
import numpy as np
import itertools

import os
import pandas as pd
import gdown
import zipfile
import moval

from moval.solvers.utils import ComputMetric, ComputAUC
from moval.models.utils import cal_softmax

"""Test the func with classification tasks.

    Examples:
        >>> pytest tests/5_test_demo_cls.py

"""

# download the data, which we used for MICCAI 2022

output = "data_moval.zip"
if not os.path.exists(output):
    url = "https://drive.google.com/u/0/uc?id=139pqxkG2ccIFq6qNArnFJWQ2by2Spbxt&export=download"
    output = "data_moval.zip"
    gdown.download(url, output, quiet=False)

directory_data = "data_moval"
if not os.path.exists(directory_data):
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(directory_data)

# now I am playing with cifar10 classification
val_data =  "data_moval/cifar10results/predictions_val.csv"
test_data = "data_moval/cifar10results/predictions_val_motion_blur.csv"
# validation data
cnn_pred = pd.read_csv(val_data)
targets_all = np.array(cnn_pred[['target_0', 'target_1', 'target_2', 'target_3', 'target_4', 
                                 'target_5', 'target_6', 'target_7', 'target_8', 'target_9']])
logits = np.array(cnn_pred[['logit_0', 'logit_1', 'logit_2', 'logit_3', 'logit_4', 
                               'logit_5', 'logit_6', 'logit_7', 'logit_8', 'logit_9']])
gt = np.argmax(targets_all, axis = 1)
# test data
cnn_pred_test = pd.read_csv(test_data)
targets_all_test = np.array(cnn_pred_test[['target_0', 'target_1', 'target_2', 'target_3', 'target_4', 
                                           'target_5', 'target_6', 'target_7', 'target_8', 'target_9']])
logits_test = np.array(cnn_pred_test[['logit_0', 'logit_1', 'logit_2', 'logit_3', 'logit_4', 
                                      'logit_5', 'logit_6', 'logit_7', 'logit_8', 'logit_9']])
gt_test = np.argmax(targets_all_test, axis = 1)

# cut some test data for acceleration
gt_test = gt_test[:5000, ]
logits_test = logits_test[:5000, :]

# logits is of shape ``(n, d)``
# gt is of shape ``(n, )``

results_files = "results_demo_cls.txt"
# clean previous results
if os.path.isfile(results_files):
    os.remove(results_files)

# estim_algorithm = "ts-model"
# mode = "classification"
# metric = "precision"
# numclass = 10
# confidence_scores = "max_class_probability-conf"
# class_specific = True
@pytest.mark.parametrize(
        "estim_algorithm, mode, metric, confidence_scores, class_specific", 
        list(itertools.product(moval.models.get_estim_options(),
                               ["classification"],
                               ["accuracy", "sensitivity", "precision", "f1score", "auc"],
                               moval.models.get_conf_options(),
                               [False, True])),
)
def test_cls(estim_algorithm, mode, metric, confidence_scores, class_specific):

    moval_model = moval.MOVAL(
                mode = mode,
                metric = metric,
                confidence_scores = confidence_scores,
                estim_algorithm = estim_algorithm,
                class_specific = class_specific
                )
    #
    moval_model.fit(logits, gt)
    estim_metric_val = moval_model.estimate(logits)
    pred = np.argmax(logits, axis = 1)

    if metric == "accuracy":
        real_metric_val = np.sum(gt == pred) / len(gt)
    elif metric == "sensitivity":
        real_sensitivities = []
        for kcls in range(logits.shape[1]):
            _, real_sensitivity, _ = ComputMetric(gt == kcls, pred == kcls)
            real_sensitivities.append(real_sensitivity)
        real_metric_val = np.mean(real_sensitivities)
    elif metric == "precision":
        real_precisions = []
        for kcls in range(logits.shape[1]):
            _, _, real_precision = ComputMetric(gt == kcls, pred == kcls)
            real_precisions.append(real_precision)
        real_metric_val = np.mean(real_precisions)
    elif metric == "f1score":
        real_F1scores = []
        for kcls in range(logits.shape[1]):
            real_F1score, _, _ = ComputMetric(gt == kcls, pred == kcls)
            real_F1scores.append(real_F1score)
        real_metric_val = np.mean(real_F1scores)
    else:
        real_auc = ComputAUC(gt, cal_softmax(logits))
        real_metric_val = np.mean(real_auc)

    err_val = np.abs( real_metric_val - np.mean(estim_metric_val) )

    # save the test err in the result files.

    pred_test = np.argmax(logits_test, axis = 1)
    
    estim_metric = moval_model.estimate(logits_test)

    if metric == "accuracy":
        real_metric = np.sum(gt_test == pred_test) / len(gt_test)
    elif metric == "sensitivity":
        real_sensitivities = []
        for kcls in range(logits.shape[1]):
            _, real_sensitivity, _ = ComputMetric(gt_test == kcls, pred_test == kcls)
            real_sensitivities.append(real_sensitivity)
        real_metric = np.mean(real_sensitivities)
    elif metric == "precision":
        real_precisions = []
        for kcls in range(logits.shape[1]):
            _, _, real_precision = ComputMetric(gt_test == kcls, pred_test == kcls)
            real_precisions.append(real_precision)
        real_metric = np.mean(real_precisions)
    elif metric == "f1score":
        real_F1scores = []
        for kcls in range(logits.shape[1]):
            real_F1score, _, _ = ComputMetric(gt_test == kcls, pred_test == kcls)
            real_F1scores.append(real_F1score)
        real_metric = np.mean(real_F1scores)
    else:
        real_auc = ComputAUC(gt_test, cal_softmax(logits_test))
        real_metric = np.mean(real_auc)

    test_condition = f"estim_algorithm = {estim_algorithm}, mode = {mode}, metric = {metric}, confidence_scores = {confidence_scores}, class_specific = {class_specific}"

    with open(results_files, 'a') as f:
        f.write(test_condition)
        f.write('\n')
        f.write("validation err: ")
        f.write(str(err_val))
        f.write('\n')
        f.write("moval parameter: ")
        f.write(str(moval_model.model_.param))
        f.write('\n')
        if moval_model.model_.extend_param:
            f.write("moval extended parameter: ")
            f.write(str(moval_model.model_.param_ext))
            f.write('\n')
        f.write("real metric: ")
        f.write(str(real_metric))
        f.write('\n')
        f.write("estimated metric: ")
        f.write(str(np.mean(estim_metric)))
        f.write('\n')
        f.write('\n')

    
