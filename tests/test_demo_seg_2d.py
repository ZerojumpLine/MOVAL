import pytest
import numpy as np
import nibabel as nib
import itertools
from moval.solvers.utils import ComputMetric

import os
import zipfile
import gdown
import moval

"""Test the func with 2d segmentation tasks.

    Examples:
        >>> pytest tests/test_demo_seg_2d.py

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

# now I am playing with prostate segmentation
Datafile_eval = "data_moval/Prostateresults/seg-eval.txt"
Imglist_eval = open(Datafile_eval)
Imglist_eval_read = Imglist_eval.read().splitlines()

logits = []
gt = []
# to accelerate the debugging speed, crop the middel 60 x 60 x 30 cub for training/inference.
for Imgname_eval in Imglist_eval_read:
    GT_file = Imgname_eval.replace("data", "data_moval")
    caseID = Imgname_eval.split("/")[-1][:6]
    logit_cls0_file = "data_moval/Prostateresults/prostateval/results/pred_" + caseID + "cls0_prob.nii.gz"
    logit_cls1_file = "data_moval/Prostateresults/prostateval/results/pred_" + caseID + "cls1_prob.nii.gz"
    logit_cls0_read = nib.load(logit_cls0_file)
    logit_cls1_read = nib.load(logit_cls1_file)
    logit_cls0      = logit_cls0_read.get_fdata()   # ``(H, W, D)``
    logit_cls1      = logit_cls1_read.get_fdata()
    GT_read         = nib.load(GT_file)
    GTimg           = GT_read.get_fdata()           # ``(H, W, D)``
    logit_cls0      = logit_cls0[logit_cls0.shape[0] //2 - 30: logit_cls0.shape[0] //2 + 30,
                                 logit_cls0.shape[1] //2 - 30: logit_cls0.shape[1] //2 + 30,
                                 logit_cls0.shape[2] //2 - 15: logit_cls0.shape[2] //2 + 15]
    logit_cls1      = logit_cls1[logit_cls1.shape[0] //2 - 30: logit_cls1.shape[0] //2 + 30,
                                 logit_cls1.shape[1] //2 - 30: logit_cls1.shape[1] //2 + 30,
                                 logit_cls1.shape[2] //2 - 15: logit_cls1.shape[2] //2 + 15]
    GTimg           = GTimg[GTimg.shape[0] //2 - 30: GTimg.shape[0] //2 + 30,
                            GTimg.shape[1] //2 - 30: GTimg.shape[1] //2 + 30,
                            GTimg.shape[2] //2 - 15: GTimg.shape[2] //2 + 15]
    logit_cls = np.stack((logit_cls0, logit_cls1))  # ``(d, H, W, D)``
    # only including the slices that contains labels
    for dslice in range(GTimg.shape[2]):
        if np.sum(GTimg[:, :, dslice]) > 0:
            logits.append(logit_cls[:, :, :, dslice])
            gt.append(GTimg[:, :, dslice])

# logits is a list of length ``n``,  each element has ``(d, H, W)``. 
# gt is a list of length ``n``,  each element has ``(H, W)``.
# H and W could differ for different cases.

Datafile_test = "data_moval/Prostateresults/seg-testA.txt"
Imglist_test = open(Datafile_test)
Imglist_test_read = Imglist_test.read().splitlines()

logits_test = []
gt_test = []
for Imgname_test in Imglist_test_read:
    GT_file = Imgname_test.replace("data", "data_moval")

    caseID = Imgname_test.split("/")[-1][:6]

    logit_cls0_file = "data_moval/Prostateresults/prostattestcondition_A/results/pred_" + caseID + "cls0_prob.nii.gz"
    logit_cls1_file = "data_moval/Prostateresults/prostattestcondition_A/results/pred_" + caseID + "cls1_prob.nii.gz"

    logit_cls0_read = nib.load(logit_cls0_file)
    logit_cls1_read = nib.load(logit_cls1_file)
    logit_cls0      = logit_cls0_read.get_fdata()
    logit_cls1      = logit_cls1_read.get_fdata()
    GT_read         = nib.load(GT_file)
    GTimg           = GT_read.get_fdata()           # ``(H, W, D)``

    logit_cls = np.stack((logit_cls0, logit_cls1))  # ``(n, H, W, D)``
    
    # only including the slices that contains labels
    for dslice in range(GTimg.shape[2]):
        if np.sum(GTimg[:, :, dslice]) > 0:
            logits_test.append(logit_cls[:, :, :, dslice])
            gt_test.append(GTimg[:, :, dslice])

# logits_test is a list of length ``n``,  each element has ``(d, H, W)``. 
# gt_test is a list of length ``n``,  each element has ``(H, W)``.
# H and W could differ for different cases.

results_files = "results_demo_seg_2d.txt"
# clean previous results
if os.path.isfile(results_files):
    os.remove(results_files)

# estim_algorithm = "ts-model"
# mode = "segmentation"
# confidence_scores = "max_class_probability-conf"
# class_specific = True

@pytest.mark.parametrize(
        "estim_algorithm, mode, confidence_scores, class_specific", 
        list(itertools.product(moval.models.get_estim_options(),
                               ["segmentation"],
                               moval.models.get_conf_options(),
                               [False, True])),
)
def test_seg_3d(estim_algorithm, mode, confidence_scores, class_specific):

    moval_model = moval.MOVAL(
                mode = mode,
                confidence_scores = confidence_scores,
                estim_algorithm = estim_algorithm,
                class_specific = class_specific
                )

    #
    moval_model.fit(logits, gt)
    estim_dsc = moval_model.estimate(logits)
    
    DSC_list = []
    for n_case in range(len(logits)):
        pred_case   = np.argmax(logits[n_case], axis = 0) # ``(H, W, (D))``
        gt_case     = gt[n_case] # ``(H, W, (D))``

        DSC = ComputMetric(pred_case == 1, gt_case == 1)
        DSC_list.append(DSC)
    m_DSC = np.mean(np.array(DSC_list))
    
    err_val_dsc = np.abs( m_DSC - estim_dsc )

    # save the test err in the result files.

    estim_dsc_test = moval_model.estimate(logits_test)

    DSC_list_test = []
    for n_case in range(len(logits_test)):
        pred_case   = np.argmax(logits_test[n_case], axis = 0) # ``(H, W, (D))``
        gt_case     = gt_test[n_case] # ``(H, W, (D))``

        DSC = ComputMetric(pred_case == 1, gt_case == 1)
        DSC_list_test.append(DSC)
    m_DSC_test = np.mean(np.array(DSC_list_test))

    err_test = np.abs( m_DSC_test - estim_dsc_test )

    test_condition = f"estim_algorithm = {estim_algorithm}, mode = {mode}, confidence_scores = {confidence_scores}, class_specific = {class_specific}"

    with open(results_files, 'a') as f:
        f.write(test_condition)
        f.write('\n')
        f.write("validation dsc err: ")
        f.write(str(err_val_dsc))
        f.write('\n')
        f.write("validation predicted dsc: ")
        f.write(str(estim_dsc))
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