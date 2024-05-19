from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np

def cal_softmax(x: np.ndarray, T = 1, e1 = 1e-6) -> np.ndarray:
    """Compute softmax values for each sets of scores in x.
    
    Args:
        x: The network output (logits) of shape ``(n, d)``.
        T: A scarlar temperature to calibrate the score.
        e1: A small number to prevent unexpected results of division.
    
    Returns:
        prob: The calculated softmax probability for different classes of shape ``(n, d)``

    """
    # to avoid exponent exploding
    x_max = np.max(x, keepdims=True, axis = 1).repeat(x.shape[1], axis=1)

    prob = np.exp((x - x_max).transpose() / T) / (np.sum(np.exp((x - x_max).transpose() / T), axis=0) + e1)

    return prob.transpose()

def cal_energy(x: np.ndarray, T = 1) -> np.ndarray:
    """Compute energy values for each sets of scores in x.

    Calculating based on the defination as described in:
    ``https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html``
    
    Args:
        x: The network output (logits) of shape ``(n, d)``.
        T: A scarlar temperature to calibrate the score.
    
    Returns:
        energy: The calculated energy of shape ``(n, )``

    """

    denominator = np.sum(np.exp(x.transpose() / T), axis=0)
    energy = - T * np.log(denominator)

    return energy

def cal_mcp(x: np.ndarray, T = 1) -> np.ndarray:
    """Compute maximum class probability (MCP) for each sets of scores in x.
    
    Args:
        x: The network output (logits) of shape ``(n, d)``.
        T: A scarlar temperature to calibrate the score.
    
    Returns:
        MCP: The calculated MCP of shape ``(n, )``

    """
    p = cal_softmax(x, T = T)
    MCP = np.max(p, axis=1)

    return MCP

def cal_entropy(x: np.ndarray, T = 1, e1 = 1e-6) -> np.ndarray:
    """Compute entropy values for each sets of scores in x.

    Note:
        This entropy calculation chooses base as the number of class to ensure normalization.
    
    Args:
        x: The network output (logits) of shape ``(n, d)``.
        T: A scarlar temperature to calibrate the score.
        e1: A small number to prevent unexpected results of division.
    
    Returns:
        entropy: The calculated entropy of shape ``(n, )``

    """
    num_class = x.shape[-1]
    p = cal_softmax(x, T = T)
    entropy = - np.sum(p * np.emath.logn(num_class, p + e1), axis=1)

    return entropy

def cal_doctor(x: np.ndarray, T = 1) -> np.ndarray:
    """Compute engery values for each sets of scores in x.

    Calculating based on the defination as described in:
    ``https://arxiv.org/abs/2106.02395``
    
    Args:
        x: The network output (logits) of shape ``(n, d)``.
        T: A scarlar temperature to calibrate the score.
    
    Returns:
        doctor: The calculated doctor of shape ``(n, )``

    """

    p = cal_softmax(x, T = T)
    g = np.sum(p ** 2, axis=1)
    doctor = (1 - g) / g

    return doctor

def sum_tensor(inp: np.ndarray, axes: np.ndarray, keepdims=False) -> np.ndarray:
    axes = np.unique(axes).astype(int)
    if keepdims:
        for ax in axes:
            inp = inp.sum(int(ax), keepdims=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output: np.ndarray, gt: np.ndarray, axes=None, square=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the true positive (tp), false positive (fp) and false negative (fn).

    Args:
        net_output: The predicted probability of shape ``(n, d, (H), (W), (D))``.
        gt: The manual label of shape ``(n, (H), (W), (D))``.
        axes: We average the axes of features. This is useful if we want to return case-wise soft dice score.
        square: Indicate if we want to include the square term in tp etc, which could be more stable.
    
    Returns:
        tp: The calculated class-wise true positive of shape ``(n, d)`` or ``(d, )``.
        fp: The calculated class-wise false positive of shape ``(n, d)`` or ``(d, )``.
        fn: The calculated class-wise false negative of shape ``(n, d)`` or ``(d, )``.

    """
    if axes is None:
        axes = tuple(range(2, len(net_output.shape)))

    shp_x = net_output.shape
    shp_y = gt.shape

    y_onehot = one_hot_embedding(gt, shp_x[1])

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdims=False)
    fp = sum_tensor(fp, axes, keepdims=False)
    fn = sum_tensor(fn, axes, keepdims=False)

    return tp, fp, fn

def SoftDiceLoss(x: np.ndarray, y: np.ndarray, smooth = 1e-5) -> np.ndarray:
    """Calculation of soft dice score / F1 score.

    Args:
        x: The predicted probability of shape ``(n, d, (H), (W), (D))``.
        y: The manual label of shape ``(n, (H), (W), (D))``.
    
    Returns:
        dice_score: Average class-wise dice of shape ``(d, )``

    """
    assert len(x.shape) == len(y.shape) + 1

    shp_x = x.shape
    square = False

    axes = [0] + list(range(2, len(shp_x)))
    tp, fp, fn = get_tp_fp_fn(x, y, axes, square)
    dc = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    dc_process = dc

    return dc_process

def SoftSensitivity(x: np.ndarray, y: np.ndarray, smooth = 1e-5) -> np.ndarray:
    """Calculation of soft sensitivity.

    Args:
        x: The predicted probability of shape ``(n, d, (H), (W), (D))``.
        y: The manual label of shape ``(n, (H), (W), (D))``.
    
    Returns:
        sensitivity_score: Average class-wise sensitivity of shape ``(d, )``

    """
    assert len(x.shape) == len(y.shape) + 1

    shp_x = x.shape
    square = False

    axes = [0] + list(range(2, len(shp_x)))
    tp, fp, fn = get_tp_fp_fn(x, y, axes, square)
    sensitivity = (tp + smooth) / (tp + fn + smooth)

    return sensitivity

def SoftPrecision(x: np.ndarray, y: np.ndarray, smooth = 1e-5) -> np.ndarray:
    """Calculation of soft precision score.

    Args:
        x: The predicted probability of shape ``(n, d, (H), (W), (D))``.
        y: The manual label of shape ``(n, (H), (W), (D))``.
    
    Returns:
        precision_score: Average class-wise precision of shape ``(d, )``

    """
    assert len(x.shape) == len(y.shape) + 1

    shp_x = x.shape
    square = False

    axes = [0] + list(range(2, len(shp_x)))
    tp, fp, fn = get_tp_fp_fn(x, y, axes, square)
    precision = (tp + smooth) / (tp + fp + smooth)

    return precision

def SoftAUC(x: np.ndarray, sel_cls: int = None) -> np.ndarray:
    """Calculate the estimated AUC given the probability.
    
    Note:
        When dealing with multi-class AUC, we calculate the One-vs-Rest strategy for simplicity.
        More information could refer to https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html.
        Similar strategy is adopted in https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html.
    
    Args:
        x: The predicted probability of shape ``(n, d, (H), (W), (D))``.
        sel_cls: Selected class to calculate AUC.

    Returns:
        AUCs: Estimated class-wise AUC of shape ``(d, )``, or ``(1, )`` if sel_cls is givien.
        TPRall: A list of d true positive rates for different classes, or a list of 1 if sel_cls is givien.
        FPRall: A list of d false positive rates for different classes, or a list of 1 if sel_cls is givien.

    """
    
    # thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.05, 
    #               0.1, 0.2, 0.3, 0.4, 0.5, 
    #               0.6, 0.7, 0.8, 0.9, 0.95, 
    #               0.99, 0.999, 0.9999, 0.99999]
    
    y = np.argmax(x, axis = 1) # this is not important, just a placeholder of shape ``(n, (H), (W), (D))``
    shp_x = x.shape
    square = False
    axes = [0] + list(range(2, len(shp_x)))
    #
    AUCs = []
    TPRall = []
    FPRall = []
    smooth = 1e-6
    #
    if sel_cls is None:
        allcls = range(x.shape[1])
    else:
        allcls = range(sel_cls, sel_cls+1)

    for test_cls in allcls:
        # low thres -> high FPR -> high sensitivity (TPR)
        # obtain TPRs and FPRs

        # customize thresholds
        # generate thresholds that fit the probability, just in case tpr == 0
        prob_max = np.min((np.max(x[:, test_cls]) - 0.05, 1.))
        prob_min = np.max((np.min(x[:, test_cls]) + 0.05, 0.))
        num_points = 5 # this might is too small, but I need to accelerate the optimization process.

        interval = (prob_max - prob_min) / (num_points - 1)
        thresholds = [prob_min + i * interval for i in range(num_points)]

        TPRs = []
        FPRs = []
        #
        TPRs.append(0.)
        FPRs.append(0.)
        for thres in thresholds:
            cls_choice = list(range(x.shape[1]))
            cls_choice.remove(test_cls)
            y[y == test_cls] = cls_choice[0] # remove all samples that predicted as ``test_cls``
            #
            mask = x[:, test_cls] > thres
            y[mask, ...] = test_cls # overwrite the prediction based on given threshold
            #
            # I only need to make sure y[y_cls == True] is correct, and ``test_cls`` does not appear otherwhere

            tp, fp, fn = get_tp_fp_fn(x, y, axes, square)

            TPR = tp[test_cls] / (np.sum(y == test_cls) + smooth)
            FPR = fp[test_cls] / (np.sum(y != test_cls) + smooth)

            TPRs.append(TPR)
            FPRs.append(FPR)
        TPRs.append(1.)
        FPRs.append(1.)
        # calculate AUC
        AUC = 0
        for k_thres in range(len(TPRs) - 1):
            TPR_left = TPRs[k_thres]
            TPR_right = TPRs[k_thres + 1]
            FPR_left = FPRs[k_thres]
            FPR_right = FPRs[k_thres + 1]
            AUC_k = (FPR_right - FPR_left) * (TPR_left + TPR_right) / 2
            AUC = AUC + AUC_k
        AUCs.append(AUC)
        TPRall.append(TPRs)
        FPRall.append(FPRs)
    
    return np.array(AUCs), TPRall, FPRall


def one_hot_embedding(labels: np.ndarray, num_class: int) -> np.ndarray:
    """Embedding labels to one-hot form.

    Args:
        labels: The label map of shape ``(n, )`` or ``(n, H, W, (D))``.
        num_class: An int scalar number of classes.

    Returns:
        labels_onehot: The transformed onehot label map of shape ``(n, d)`` or ``(n, d, H, W, (D))``.
    
    """
    y = np.eye(num_class)[labels.reshape(-1).astype(int)]  # of shape `(n * H * W (*D), d)`
    labels_onehot = y.reshape(list(labels.shape)+[num_class]) # of shape `(n, H, W, (D), d)`

    axes = [0] + [len(labels.shape)] + list(range(1, len(labels.shape)))
    labels_onehot = np.transpose(labels_onehot, axes) # of shape `(n, d, H, W, (D))`

    return labels_onehot