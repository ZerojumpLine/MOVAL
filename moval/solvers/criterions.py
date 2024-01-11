import abc
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
from moval.solvers import register
from moval.models.utils import cal_softmax
from moval.solvers.utils import ComputMetric, ComputAUC

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
    def __init__(self, class_specific=False, metric="accuracy"):
        super().__init__()
        self.class_specific = class_specific
        self.metric = metric
    
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

            if self.metric == "accuracy":
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(pred == kcls)[0]
                    # if there do not exist any samplies for class kcls
                    if len(pos_cls) == 0:
                        acc_cls = np.sum(gt == pred) / len(gt)
                    else:
                        acc_cls = np.sum(gt[pos_cls] == pred[pos_cls]) / len(gt[pos_cls])
                    err[kcls] = estim[kcls] - acc_cls

            elif self.metric == "sensitivity":
                sensitivities = []
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(gt == kcls)[0]
                    # if there do not exist any samplies for class kcls
                    if len(pos_cls) == 0:
                        sensitivity_cls = -1
                    else:
                        _, sensitivity_cls, _ = ComputMetric(gt == kcls, pred == kcls)
                    sensitivities.append(sensitivity_cls)
                sensitivities = np.array(sensitivities)
                sensitivity_mean = sensitivities[sensitivities >= 0].mean()
                sensitivities[sensitivities < 0] = sensitivity_mean

                err = estim - sensitivities

            elif self.metric == "precision":
                precisions = []
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(gt == kcls)[0]
                    # if there do not exist any samplies for class kcls
                    if len(pos_cls) == 0:
                        precision_cls = -1
                    else:
                        _, _, precision_cls = ComputMetric(gt == kcls, pred == kcls)
                    precisions.append(precision_cls)
                precisions = np.array(precisions)
                precision_mean = precisions[precisions >= 0].mean()
                precisions[precisions < 0] = precision_mean

                err = estim - precisions

            elif self.metric == "f1score":
                f1scores = []
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(gt == kcls)[0]
                    # if there do not exist any samplies for class kcls
                    if len(pos_cls) == 0:
                        f1score_cls = -1
                    else:
                        f1score_cls, _, _ = ComputMetric(gt == kcls, pred == kcls)
                    f1scores.append(f1score_cls)
                f1scores = np.array(f1scores)
                f1score_mean = f1scores[f1scores >= 0].mean()
                f1scores[f1scores < 0] = f1score_mean

                err = estim - f1scores

            elif self.metric == "auc":
                aucs = ComputAUC(gt, cal_softmax(inp))
                auc_mean = aucs[aucs > 0].mean()
                aucs[aucs == 0] = auc_mean

                err = estim - aucs

            else:
                ValueError(f"Unsupported metric '{self.metric}'")

        return np.abs(err)

@register("seg-calibrate")
class segCalibrate(Calibrate):
    def __init__(self, class_specific=False, metric="accuracy"):
        super().__init__()
        self.class_specific = class_specific
        self.metric = metric
    
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
            err: The difference between the estimated metric and the real metric.
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
            
            if self.metric == "accuracy":
                
                err = np.zeros(inp[0].shape[0])
                for kcls in range(0, inp[0].shape[0]):
                    
                    pred_all_flatten_kcls = []
                    gt_all_flatten_kcls = []

                    for n_case in range(len(inp)):

                        pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                        gt_case     = gt[n_case] # ``(H, W, (D))``
                        pos_kcls = np.where(pred_case.flatten() == kcls)[0]
                        pred_all_flatten_kcls.append(pred_case.flatten()[pos_kcls])
                        gt_all_flatten_kcls.append(gt_case.flatten()[pos_kcls])
                    
                    pred_all_flatten_kcls = np.concatenate(pred_all_flatten_kcls, axis=0)
                    gt_all_flatten_kcls = np.concatenate(gt_all_flatten_kcls, axis=0)

                    acc_cls = np.sum(gt_all_flatten_kcls == pred_all_flatten_kcls) / len(gt_all_flatten_kcls)
                    err[kcls] = estim[kcls] - acc_cls
                
            elif self.metric == "sensitivity":

                sensitivity = []
                for n_case in range(len(inp)):

                    pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                    gt_case     = gt[n_case] # ``(H, W, (D))``

                    sensitivity_case = np.zeros(inp[n_case].shape[0])
                    for kcls in range(1, inp[n_case].shape[0]):
                        if np.sum(gt_case == kcls) == 0:
                            sensitivity_case[kcls] = -1
                        else:
                            _, sensitivity_case[kcls], _ = ComputMetric(gt_case == kcls, pred_case == kcls)
                    sensitivity.append(sensitivity_case)
                
                # only aggregate the ones which are not -1
                sensitivity = np.array(sensitivity) # ``(n, d)``
                sensitivity_mean = []
                for kcls in range(inp[n_case].shape[0]):
                    # I am not sure, if the real sensitivity is 0, I think the network cannot learn anything
                    # but any calculate this case in the estim, may we can improve here.
                    sensitivity_mean.append(sensitivity[:, kcls][sensitivity[:,kcls] >= 0].mean())
                sensitivity_mean = np.array(sensitivity_mean)

                err = estim - sensitivity_mean

            elif self.metric == "precision":

                precision = []
                for n_case in range(len(inp)):

                    pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                    gt_case     = gt[n_case] # ``(H, W, (D))``

                    precision_case = np.zeros(inp[n_case].shape[0])
                    for kcls in range(1, inp[n_case].shape[0]):
                        if np.sum(gt_case == kcls) == 0:
                            precision_case[kcls] = -1
                        else:
                            _, _, precision_case[kcls] = ComputMetric(gt_case == kcls, pred_case == kcls)
                    precision.append(precision_case)
                
                # only aggregate the ones which are not -1
                precision = np.array(precision) # ``(n, d)``
                precision_mean = []
                for kcls in range(inp[n_case].shape[0]):
                    # I am not sure, if the real precision is 0, I think the network cannot learn anything
                    # but any calculate this case in the estim, may we can improve here.
                    precision_mean.append(precision[:, kcls][precision[:,kcls] >= 0].mean())
                precision_mean = np.array(precision_mean)

                err = estim - precision_mean

            elif self.metric == "f1score":

                dsc = []
                for n_case in range(len(inp)):

                    pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                    gt_case     = gt[n_case] # ``(H, W, (D))``

                    dsc_case = np.zeros(inp[n_case].shape[0])
                    for kcls in range(1, inp[n_case].shape[0]):
                        if np.sum(gt_case == kcls) == 0:
                            dsc_case[kcls] = -1
                        else:
                            dsc_case[kcls], _, _ = ComputMetric(gt_case == kcls, pred_case == kcls)
                    dsc.append(dsc_case)
                
                # only aggregate the ones which are not -1
                dsc = np.array(dsc) # ``(n, d)``
                dsc_mean = []
                for kcls in range(inp[n_case].shape[0]):
                    # I am not sure, if the real dsc is 0, I think the network cannot learn anything
                    # but any calculate this case in the estim, may we can improve here.
                    dsc_mean.append(dsc[:, kcls][dsc[:,kcls] >= 0].mean())
                dsc_mean = np.array(dsc_mean)

                err = estim - dsc_mean

            elif self.metric == "auc":
                
                auc = []
                for n_case in range(len(inp)):
                    
                    inp_case = inp[n_case] # ``(d, H, W, (D))``
                    # from ``(d, H, W, (D))`` to ``(n, d)``
                    d, *rest_of_dimensions = inp_case.shape
                    flatten_dim = np.prod(rest_of_dimensions)
                    inp_case = inp_case.reshape((d, flatten_dim))
                    inp_case = inp_case.T # ``(n, d)``
                    probability = cal_softmax(inp_case) # ``(n, d)``                    
                    gt_case     = gt[n_case].flatten() # ``(n, )``

                    auc_case = ComputAUC(gt_case, probability) # ``(d, )``
                    auc.append(auc_case)
                
                # only aggregate the ones which are not -1
                auc = np.array(auc) # ``(n, d)``
                auc_mean = []
                for kcls in range(inp[n_case].shape[0]):
                    # I am not sure, if the real dsc is 0, I think the network cannot learn anything
                    # but any calculate this case in the estim, may we can improve here.
                    auc_mean.append(auc[:, kcls][auc[:,kcls] >= 0].mean())
                auc_mean = np.array(auc_mean)

                err = estim - auc_mean

            else:
                ValueError(f"Unsupported metric '{self.metric}'")

            return np.abs(err)