import abc
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union
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
        inp: Union[List[Iterable], np.ndarray],
        gt: Union[List[Iterable], np.ndarray],
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
        inp: List[Iterable],
        gt: List[Iterable],
        estim: np.ndarray
    ) -> np.ndarray:
        """Compute the calibration err for segmentation tasks.
        
        Args:
            inp: The network output (logits) as a list of n ``(d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation as a list of n ``(H, W, (D))`` for segmentation. 
            estim: The estimated performance by moval.
                For class-specific version, estim is of shape ``(d, )`` which contains differences for different classes. 
        
        Returns:
            err: The difference between the estimated DSC and the real DSC.
                For class-specific version, err is of shape ``(d, )`` which contains differences for different classes.

        """

        if not self.class_specific:
            pred_all_flatten = []
            gt_all_flatten = []
            for n_case in range(len(inp)):

                pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                gt_case     = gt[n_case] # ``(H, W, (D))``
                
                pred_all_flatten.append(pred_case.flatten())
                gt_all_flatten.append(gt_case.flatten())
            
            pred_all_flatten = np.concatenate(pred_all_flatten)
            gt_all_flatten = np.concatenate(gt_all_flatten)

            acc = np.sum(gt_all_flatten == pred_all_flatten) / len(gt_all_flatten)
            err = estim - acc

            return np.abs(err)

        else:
            
            pred_all_flatten_bg = []
            gt_all_flatten_bg = []
            dsc = []

            for n_case in range(len(inp)):

                pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                gt_case     = gt[n_case] # ``(H, W, (D))``
                
                pos_bg = np.where(pred_case.flatten() == 0)[0]

                pred_all_flatten_bg.append(pred_case.flatten()[pos_bg])
                gt_all_flatten_bg.append(gt_case.flatten()[pos_bg])

                dsc_case = np.zeros(inp[n_case].shape[0])
                for kcls in range(1, inp[n_case].shape[0]):
                    dsc_case[kcls] = ComputMetric(pred_case == kcls, gt_case == kcls)
                dsc.append(dsc_case)

            pred_all_flatten_bg = np.concatenate(pred_all_flatten_bg)
            gt_all_flatten_bg = np.concatenate(gt_all_flatten_bg)

            acc_bg = np.sum(gt_all_flatten_bg == pred_all_flatten_bg) / len(gt_all_flatten_bg)
            
            err = estim - np.mean(np.array(dsc), axis=0)
            err[0] = estim[0] - acc_bg

            return np.abs(err)