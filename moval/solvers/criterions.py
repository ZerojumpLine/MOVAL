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
        estim: List[Iterable],
        kcls_all: List[Iterable] = None
    ) -> np.ndarray:
        """Compute the calibration err for classification tasks.
        
        Args:
            inp: The network output (logits) of shape ``(n, d)``. 
            gt: The cooresponding annotation of shape ``(n, )``.
            estim: The estimated performance by moval, this is a float for class agnostic version.
                For class-specific version, estim is list of shape ``(1, )`` which contains estimation for kcls. 
            kcls_all: This contains the class list, only for the class-specific verision, which indicates the class to be optimized.
        
        Returns:
            err: The difference between the estimated accuracy and the real accuracy.
                For class-specific version, err contains differences for class kcls.
        
        """

        pred = np.argmax(inp, axis = 1)

        if not self.class_specific:
            
            acc = np.sum(gt == pred) / len(gt)
            err = estim - acc

        else:
            w_kcls_all = []
            for kcls in kcls_all:
                pos_cls = np.where(pred == kcls)[0]
                w_kcls_all.append(len(pos_cls))        
            w_kcls_all = w_kcls_all / np.sum(w_kcls_all) # normalize
            err = 0
            k_cnt = 0
            if self.metric == "accuracy":
                e1 = 1e-6 # small number to avoid nan.
                for kcls in kcls_all:
                    pos_cls = np.where(pred == kcls)[0]
                    acc_cls = np.sum(gt[pos_cls] == pred[pos_cls]) / (len(gt[pos_cls]) + e1)
                    err += w_kcls_all[k_cnt] * (estim[k_cnt] - acc_cls)
                    k_cnt += 1

            elif self.metric == "sensitivity":
                for kcls in kcls_all:
                    _, sensitivity_cls, _ = ComputMetric(gt == kcls, pred == kcls)
                    err += w_kcls_all[k_cnt] * (estim[k_cnt] - sensitivity_cls)
                    k_cnt += 1

            elif self.metric == "precision":
                for kcls in kcls_all:
                    _, _, precision_cls = ComputMetric(gt == kcls, pred == kcls)
                    err += w_kcls_all[k_cnt] * (estim[k_cnt] - precision_cls)
                    k_cnt += 1

            elif self.metric == "f1score":
                for kcls in kcls_all:
                    f1score_cls, _, _ = ComputMetric(gt == kcls, pred == kcls)
                    err += w_kcls_all[k_cnt] * (estim[k_cnt] - f1score_cls)
                    k_cnt += 1

            elif self.metric == "auc":
                for kcls in kcls_all:
                    aucs = ComputAUC(gt, cal_softmax(inp), kcls)
                    err += w_kcls_all[k_cnt] * (estim[k_cnt] - aucs)
                    k_cnt += 1

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
        estim: List[Iterable],
        kcls_all: List[Iterable] = None
    ) -> np.ndarray:
        """Compute the calibration err for segmentation tasks.
        
        Args:
            inp: The network output (logits) as a list of n ``(d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation as a list of n ``(H, W, (D))`` for segmentation. 
            estim: The estimated performance by moval, this is a float for class agnostic version.
                For class-specific version, estim is list of shape ``(1, )`` which contains estimation for kcls. 
            kcls_all: This contains the class list, only for the class-specific verision, which indicates the class to be optimized.
        
        Returns:
            err: The difference between the estimated metric and the real metric.
                For class-specific version, err contains differences for class kcls.

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
            estim_kcls = estim[0]
            kcls = kcls_all[0]

            if self.metric == "accuracy":
                
                err = np.zeros(inp[0].shape[0])
                    
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
                err = estim_kcls - acc_cls
                
            elif self.metric == "sensitivity":

                sensitivity = []
                for n_case in range(len(inp)):

                    pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                    gt_case     = gt[n_case] # ``(H, W, (D))``

                    if np.sum(gt_case == kcls) == 0:
                        sensitivity_cls = -1
                    else:
                        _, sensitivity_cls, _ = ComputMetric(gt_case == kcls, pred_case == kcls)
                    sensitivity.append(sensitivity_cls)
                
                # only aggregate the ones which are not -1
                sensitivity = np.array(sensitivity) # ``(n, )``
                # I am not sure, if the real sensitivity is 0, I think the network cannot learn anything
                # but any calculate this case in the estim, may we can improve here.
                sensitivity_mean = sensitivity[sensitivity >= 0].mean()

                err = estim_kcls - sensitivity_mean

            elif self.metric == "precision":

                precision = []
                for n_case in range(len(inp)):

                    pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                    gt_case     = gt[n_case] # ``(H, W, (D))``

                    if np.sum(gt_case == kcls) == 0:
                        precision_cls = -1
                    else:
                        _, _, precision_cls = ComputMetric(gt_case == kcls, pred_case == kcls)
                    precision.append(precision_cls)
                
                # only aggregate the ones which are not -1
                precision = np.array(precision) # ``(n, d)``
                # I am not sure, if the real precision is 0, I think the network cannot learn anything
                # but any calculate this case in the estim, may we can improve here.
                precision_mean = precision[precision >= 0].mean()

                err = estim_kcls - precision_mean

            elif self.metric == "f1score":

                dsc = []
                for n_case in range(len(inp)):

                    pred_case   = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                    gt_case     = gt[n_case] # ``(H, W, (D))``

                    if np.sum(gt_case == kcls) == 0:
                        dsc_cls = -1
                    else:
                        dsc_cls, _, _ = ComputMetric(gt_case == kcls, pred_case == kcls)
                    dsc.append(dsc_cls)
                
                # only aggregate the ones which are not -1
                dsc = np.array(dsc) # ``(n, d)``
                # I am not sure, if the real dsc is 0, I think the network cannot learn anything
                # but any calculate this case in the estim, may we can improve here.
                dsc_mean = dsc[dsc >= 0].mean()

                err = estim_kcls - dsc_mean

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

                    auc_case = ComputAUC(gt_case, probability, kcls) # ``(1, )``
                    auc.append(auc_case)
                
                # only aggregate the ones which are not -1
                auc = np.array(auc) # ``(n, 1)``
                # I am not sure, if the real dsc is 0, I think the network cannot learn anything
                # but any calculate this case in the estim, may we can improve here.
                auc_mean = auc[:, 0][auc[:, 0] >= 0].mean()

                err = estim_kcls - auc_mean

            else:
                ValueError(f"Unsupported metric '{self.metric}'")

            return np.abs(err)