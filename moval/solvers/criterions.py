import abc
import numpy as np
from moval.solvers import register
from moval.solvers.utils import ComputMetric

class Calibrate(abc.ABC):
    """Base model for calibration criterions

    Features can be computed by calling the ``__call__()`` method. This class should not 
    be directly instaniated, and instead used as the base class for calibration criterions.

    """

    def __call__(
        self,
        inp: np.ndarray,
        gt: np.ndarray,
        estim: np.ndarray
    ) -> np.ndarray:
        """Compute the calibration error.
        """
        raise NotImplementedError()

@register("cls-calibrate")
class clsCalibrate(Calibrate):
    def __init__(self, class_specific=False):
        super().__init__()
        self.class_specific = class_specific
    
    def __call__(
        self,
        inp: np.ndarray,
        gt: np.ndarray,
        estim: np.ndarray
    ) -> np.ndarray:
        """Compute the calibration err for classification tasks.
        
        Args:
            inp: The network output (logits) of shape ``(n, d)``. 
            gt: The cooresponding annotation of shape ``(n, )``.
            estim: The estimated performance by moval.
                For class-specific version, estim is of shape ``(d, )`` which contains differences for different classes. 
        
        Returns:
            err: The difference between the estimated accuracy and the real accuracy.
                For class-specific version, err is of shape ``(d, )`` which contains differences for different classes.
        
        """

        pred = np.argmax(inp, axis = 1)

        if not self.class_specific:
            
            acc = np.sum(gt == pred) / len(gt)
            err = estim - acc

        else:

            err = np.zeros(inp.shape[1])
            for kcls in range(inp.shape[1]):
                pos_cls = np.where(pred == kcls)[0]
                acc_cls = np.sum(gt[pos_cls] == pred[pos_cls]) / len(gt[pos_cls])
                err[kcls] = estim[kcls] - acc_cls

        return np.abs(err)

@register("seg-calibrate")
class segCalibrate(Calibrate):
    def __init__(self, class_specific=False):
        super().__init__()
        self.class_specific = class_specific
    
    def __call__(
        self,
        inp: np.ndarray,
        gt: np.ndarray,
        estim: np.ndarray
    ) -> np.ndarray:
        """Compute the calibration err for segmentation tasks.
        
        Args:
            inp: The network output (logits) of shape ``(n, d, H, W, (D))``. 
            gt: The cooresponding annotation of shape ``(n, H, W)`` or ``(n, H, W, D)``.
            estim: The estimated performance by moval.
                For class-specific version, estim is of shape ``(d, )`` which contains differences for different classes. 
        
        Returns:
            err: The difference between the estimated DSC and the real DSC.
                For class-specific version, err is of shape ``(d, )`` which contains differences for different classes.

        """

        pred = np.argmax(inp, axis = 1)

        if not self.class_specific:

            acc = np.sum(gt.flatten() == pred.flatten()) / len(gt.flatten())
            err = estim - acc

        else:
            
            err = np.zeros(inp.shape[1])
            pos_bg = np.where(pred.flatten() == 0)[0]
            acc_bg = np.sum(gt.flatten()[pos_bg] == pred.flatten()[pos_bg]) / len(gt.flatten()[pos_bg])
            err[0] = estim[0] - acc_bg

            for kcls in range(1, inp.shape[1]):
                err[kcls] = estim[kcls] - ComputMetric(pred == kcls, gt == kcls)

        return np.abs(err)