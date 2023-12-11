import pytest
import numpy as np
import nibabel as nib
import itertools
from moval.solvers.utils import ComputMetric

import os
import zipfile
import gdown
import moval

"""Test the func with 2d segmentation tasks with multiple classes.

    Examples:
        >>> pytest tests/test_demo_seg_2d_multicls.py

"""

# download the data of cardiac

output = "data_moval_supp.zip"
if not os.path.exists(output):
    url = "https://drive.google.com/u/0/uc?id=1ZlC66MGmPlf05aYYCKBaRT2q5uod8GFk&export=download"
    output = "data_moval_supp.zip"
    gdown.download(url, output, quiet=False)

directory_data = "data_moval_supp"
if not os.path.exists(directory_data):
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(directory_data)

# now I am playing with cardiac segmentation

Datafile_eval = "data_moval_supp/Cardiacresults/seg-eval.txt"
Imglist_eval = open(Datafile_eval)
Imglist_eval_read = Imglist_eval.read().splitlines()

logits = []
gt = []
for Imgname_eval in Imglist_eval_read:
    #
    caseID = Imgname_eval.split("/")[-2]
    #
    GT_file = f"data_moval_supp/Cardiacresults/GT/1/{caseID}/seg.nii.gz"
    #
    logit_cls0_file = "data_moval_supp/Cardiacresults/cardiacval/results/pred_" + caseID + "cls0_prob.nii.gz"
    logit_cls1_file = "data_moval_supp/Cardiacresults/cardiacval/results/pred_" + caseID + "cls1_prob.nii.gz"
    logit_cls2_file = "data_moval_supp/Cardiacresults/cardiacval/results/pred_" + caseID + "cls2_prob.nii.gz"
    logit_cls3_file = "data_moval_supp/Cardiacresults/cardiacval/results/pred_" + caseID + "cls3_prob.nii.gz"
    #
    logit_cls0_read = nib.load(logit_cls0_file)
    logit_cls1_read = nib.load(logit_cls1_file)
    logit_cls2_read = nib.load(logit_cls2_file)
    logit_cls3_read = nib.load(logit_cls3_file)
    #
    logit_cls0      = logit_cls0_read.get_fdata()   # ``(H, W, D)``
    logit_cls1      = logit_cls1_read.get_fdata()
    logit_cls2      = logit_cls2_read.get_fdata()
    logit_cls3      = logit_cls3_read.get_fdata()
    #
    GT_read         = nib.load(GT_file)
    GTimg           = GT_read.get_fdata()           # ``(H, W, D)``
    #
    logit_cls = np.stack((logit_cls0, logit_cls1, logit_cls2, logit_cls3))  # ``(d, H, W, D)``
    # only including the slices that contains labels
    for dslice in range(GTimg.shape[2]):
        if np.sum(GTimg[:, :, dslice]) > 0:
            logits.append(logit_cls[:, :, :, dslice])
            gt.append(GTimg[:, :, dslice])

# logits is a list of length ``n``,  each element has ``(d, H, W)``. 
# gt is a list of length ``n``,  each element has ``(H, W)``.
# H and W could differ for different cases.

Datafile_test = "data_moval_supp/Cardiacresults/seg-testA.txt"
Imglist_test = open(Datafile_test)
Imglist_test_read = Imglist_test.read().splitlines()

logits_test = []
gt_test = []
for Imgname_eval in Imglist_test_read:
    caseID = Imgname_eval.split("/")[-2]
    #
    GT_file = f"data_moval_supp/Cardiacresults/GT/2/{caseID}/seg.nii.gz"
    #
    logit_cls0_file = "data_moval_supp/Cardiacresults/cardiactest_2/results/pred_" + caseID + "cls0_prob.nii.gz"
    logit_cls1_file = "data_moval_supp/Cardiacresults/cardiactest_2/results/pred_" + caseID + "cls1_prob.nii.gz"
    logit_cls2_file = "data_moval_supp/Cardiacresults/cardiactest_2/results/pred_" + caseID + "cls2_prob.nii.gz"
    logit_cls3_file = "data_moval_supp/Cardiacresults/cardiactest_2/results/pred_" + caseID + "cls3_prob.nii.gz"
    #
    logit_cls0_read = nib.load(logit_cls0_file)
    logit_cls1_read = nib.load(logit_cls1_file)
    logit_cls2_read = nib.load(logit_cls2_file)
    logit_cls3_read = nib.load(logit_cls3_file)
    #
    logit_cls0      = logit_cls0_read.get_fdata()   # ``(H, W, D)``
    logit_cls1      = logit_cls1_read.get_fdata()
    logit_cls2      = logit_cls2_read.get_fdata()
    logit_cls3      = logit_cls3_read.get_fdata()
    #
    GT_read         = nib.load(GT_file)
    GTimg           = GT_read.get_fdata()           # ``(H, W, D)``
    logit_cls = np.stack((logit_cls0, logit_cls1, logit_cls2, logit_cls3))  # ``(d, H, W, D)``
    # only including the slices that contains labels
    for dslice in range(GTimg.shape[2]):
        if np.sum(GTimg[:, :, dslice]) > 0:
            logits_test.append(logit_cls[:, :, :, dslice])
            gt_test.append(GTimg[:, :, dslice])

# logits_test is a list of length ``n``,  each element has ``(d, H, W)``. 
# gt_test is a list of length ``n``,  each element has ``(H, W)``.
# H and W could differ for different cases.

results_files = "results_demo_seg_2d_multicls.txt"
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
                class_specific = class_specific,
                approximate = True,
                approximate_boundary = 10 # may not enough, sugguest 30, but will be slow.
                )

    #
    moval_model.fit(logits, gt)
    estim_dsc = moval_model.estimate(logits)
    
    DSC_list = []
    for n_case in range(len(logits)):
        pred_case   = np.argmax(logits[n_case], axis = 0) # ``(H, W, (D))``
        gt_case     = gt[n_case] # ``(H, W, (D))``

        DSC_c1 = ComputMetric(pred_case == 1, gt_case == 1)
        DSC_c2 = ComputMetric(pred_case == 2, gt_case == 2)
        DSC_c3 = ComputMetric(pred_case == 3, gt_case == 3)
        DSC_list.append(np.array([DSC_c1, DSC_c2, DSC_c3]))

    m_DSC = np.mean(np.array(DSC_list), axis=0)
    
    err_val_dsc = np.abs( m_DSC - estim_dsc )

    # save the test err in the result files.

    estim_dsc_test = moval_model.estimate(logits_test)

    DSC_list_test = []
    for n_case in range(len(logits_test)):
        pred_case   = np.argmax(logits_test[n_case], axis = 0) # ``(H, W, (D))``
        gt_case     = gt_test[n_case] # ``(H, W, (D))``

        DSC_c1 = ComputMetric(pred_case == 1, gt_case == 1)
        DSC_c2 = ComputMetric(pred_case == 2, gt_case == 2)
        DSC_c3 = ComputMetric(pred_case == 3, gt_case == 3)
        DSC_list_test.append(np.array([DSC_c1, DSC_c2, DSC_c3]))
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