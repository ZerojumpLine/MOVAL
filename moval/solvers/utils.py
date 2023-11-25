import numpy as np

def ComputMetric(ACTUAL: np.ndarray, PREDICTED: np.ndarray) -> float:
    """Caculate the DSC between prediction and the GT.

    Args:
        PREDICTED: The predicted segmentation map of shape ``((n), H, W, (D))`.`
        ACTUAL: The ground truth segmentation map of shape ``((n), H, W, (D))`.

    Returns:
        dice: A float scalar which represents the calculated dice score.
    
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
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
    return dice