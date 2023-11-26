import pytest
import numpy as np
import itertools

import os
import pandas as pd
import gdown
import zipfile
import moval

"""Test the func with classification tasks.

    Examples:
        >>> pytest tests/test_demo_cls.py

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

# logits is of shape ``(n, d)``
# gt is of shape ``(n, )``

results_files = "results_demo_cls.txt"
# clean previous results
if os.path.isfile(results_files):
    os.remove(results_files)

@pytest.mark.parametrize(
        "estim_algorithm, mode, confidence_scores, class_specific", 
        list(itertools.product(moval.models.get_estim_options(),
                               ["classification"],
                               moval.models.get_conf_options(),
                               [False, True])),
)
def test_cls(estim_algorithm, mode, confidence_scores, class_specific):

    moval_model = moval.MOVAL(
                mode = mode,
                confidence_scores = confidence_scores,
                estim_algorithm = estim_algorithm,
                class_specific = class_specific
                )

    #
    moval_model.fit(logits, gt)
    estim_acc = moval_model.estimate(logits)
    pred = np.argmax(logits, axis = 1)
    err_val = np.abs( np.sum(gt == pred) / len(gt) - estim_acc )

    # save the test err in the result files.

    estim_acc_test = moval_model.estimate(logits_test)
    pred_test = np.argmax(logits_test, axis = 1)
    err_test = np.abs( np.sum(gt_test == pred_test) / len(gt_test) - estim_acc_test )

    test_condition = f"estim_algorithm = {estim_algorithm}, mode = {mode}, confidence_scores = {confidence_scores}, class_specific = {class_specific}"

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
        f.write("test err: ")
        f.write(str(err_test))
        f.write('\n')
        f.write('\n')

    
