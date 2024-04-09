from sklearn.metrics import auc, roc_curve
from moval.models.utils import one_hot_embedding
import numpy as np

def ComputMetric(ACTUAL: np.ndarray, PREDICTED: np.ndarray) -> float:
    """Calculate the DSC (F1-Score), sensitivity and precision between prediction and the GT.

    Args:
        PREDICTED: The predicted segmentation map of shape ``(H, W, (D))`` or predicted classification results of shape ``(n, )``.
        ACTUAL: The ground truth segmentation map of shape ``(H, W, (D))`` or predicted classification results of shape ``(n, )``.

    Returns:
        dice: A float scalar which represents the calculated dice score.
        sensitivity: A float scalar which represents the calculated sensitivity score.
        precision: A float scalar which represents the calculated precision score.
    
    """
    ACTUAL = ACTUAL.flatten()
    PREDICTED = PREDICTED.flatten()
    idxp = ACTUAL == True
    idxn = ACTUAL == False

    tp = np.sum(ACTUAL[idxp] == PREDICTED[idxp])
    tn = np.sum(ACTUAL[idxn] == PREDICTED[idxn])
    fp = np.sum(idxn) - tn
    fn = np.sum(idxp) - tp
    if tp == 0 :
        dice = 0
        sensitivity = 0
        precision = 0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)
    return dice, sensitivity, precision

# AUC calculation here.
def ComputAUC(ACTUAL: np.ndarray, PROBABILITY: np.ndarray, sel_cls: int = None) -> float:
    """Calculate the AUC.

    Note:
        When dealing with multi-class AUC, we calculate the One-vs-Rest strategy for simplicity.
        More information could refer to https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html.
        It is not used in the framework of MOVAL, but could be useful to validate moval's performance.

    Args:
        ACTUAL: The ground truth segmentation map of shape ``(H, W, (D))`` or predicted classification results of shape ``(n, )``.
        PROBABILITY: The predicted probability of shape ``(d, H, W, (D))`` or predicted classification results of shape ``(n, d)``.
        sel_cls: The selected class for calculation. If it is None, return all classes.
    
    Returns:
        AUCs: The calculate class-wise AUC of shape ``(d, )``, or ``(1, )`` if sel_cls is givien.
    
    """

    ACTUAL = ACTUAL.flatten() # ``(n, )``
    if len(PROBABILITY.shape) > 2:
        # from ``(d, H, W, (D))`` to ``(n, d)``
        d, *rest_of_dimensions = PROBABILITY.shape
        flatten_dim = np.prod(rest_of_dimensions)
        PROBABILITY = PROBABILITY.reshape((d, flatten_dim))
        PROBABILITY = PROBABILITY.T

    y_onehot_test = one_hot_embedding(ACTUAL, PROBABILITY.shape[1])
    fpr, tpr, roc_auc = dict(), dict(), dict()

    if sel_cls is None:
        allcls = range(PROBABILITY.shape[1])
    else:
        allcls = range(sel_cls, sel_cls+1)

    for i in allcls:
        
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], PROBABILITY[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return np.array(list(roc_auc.values()))

