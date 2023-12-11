import abc
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
from typing import Any
from moval.models import register
import moval.models
from moval.models.utils import SoftDiceLoss

class Model(abc.ABC):
    """Base model for MOVAL experiments.

    Features can be computed by calling the ``__call__()`` method. This class should not 
    be directly instaniated, and instead used as the base class for MOVAl models.

    Args:
        mode (str): The given task to estimate model performance.
        num_class (int): The number of class for the given task.
        confidence_scores (str): The method to calculate confidence scores.
        class_specific (bool): If ``True``, the calculation will match class-wise confidence to class-wise accuracy/DSC.
    
    Atrributes:
        estim_algorithm (str):
            The chosen estimated algorithms for confidence calibration.
        mode (str):
            The given task to estimate model performance.
            This will choose the criterions for parameter optimization.
        num_class (int): 
            The number of class for the given task.
            This will decide the number of parameter to optimize and optimization criterions.
        confidence_scores (str): 
            The method to calculate confidence scores. 
            This will decide the methods chosen from confidences.
        class_spcific (bool):
            If ``True``, the calculation will match class-wise confidence to class-wise accuracy/DSC.
            This will also affect the number of parameter to optimize and optimization criterions.
        conf (moval.models.Confidence):
            The Confidence class to calculate the confidence scores.
        max_value (float):
            The maximum value to normalize the confidence score before calibration.
            This is only necessary when :py:attr:`conf` is unbounded.
        min_value (float):
            The minimum value to normalize the confidence score before calibration.
            This is only necessary when :py:attr:`conf` is unbounded.
        param (np.ndarray):
            The parameter to be optimized.
        extend_param (bool):
            If ``True``, the model contains more parameters (x2) and need two-stage optimization.
            Currently, this is only applicable for ``ts-atc-model``.
        is_fitted (bool):
            If ``True``, the model is fitted using validation data and ready to use for estimating performance.
        is_training (bool):
            If ``True``, the normalization paramters would be updated using the input data.

    """

    def __init__(
        self,
        estim_algorithm: str,
        mode: str,
        num_class: int,
        confidence_scores: str,
        class_specific: bool,
        extend_param: bool
    ):
        self.estim_algorithm = estim_algorithm
        self.mode = mode
        self.num_class = num_class
        self.confidence_scores = confidence_scores
        self.class_specific = class_specific
        self.is_fitted = False
        self.extend_param = extend_param

        self.conf = moval.models.init(self.confidence_scores)
        
        self.is_training = False
        
        # Note, for non-normalized method, we choose the parameter as the lower bound for normalization.
        if self.conf.normalization:
            self.max_value = 0.
            self.min_value = 0.

        if self.class_specific:
            self.param = np.ones(num_class)
        else:
            self.param = np.array([1])
        
        # allocate addtional set of parameters.
        if self.extend_param:
            if self.class_specific:
                self.param_ext = np.ones(num_class)
            else:
                self.param_ext = np.array([1])
    
    def train(self):
        """Initialize the model into training mode.
        """
        self.is_training = True
    
    def eval(self):
        """Initialize the model into evaluation mode.
        """
        self.is_training = False

    def _find_normalize(self, scores: Union[List[Iterable], np.ndarray]):
        """Find the normalization parameter for the confidence scores.

        Args:
            scores: The unbounded confidence scores, of shape ``(n, )`` or a list of n ``(H, W, (D))``.

        Notes:
            The intensity value, max and min value, will be saved.
        
        """
        if isinstance(scores, list):
            max_value_list = []
            min_value_list = []
            for score in scores:
                max_value_list.append(np.max(score))
                min_value_list.append(np.min(score))
            self.max_value = np.max(max_value_list)
            self.min_value = np.min(min_value_list)
        
        else:
            self.max_value = np.max(scores)
            self.min_value = np.min(scores)

    def _normalize(self, scores: Union[List[Iterable], np.ndarray], lb: np.ndarray = None, pred: np.ndarray = None, e1: float = 1e-6) -> Union[List[Iterable], np.ndarray]:
        """Normalize the confidence scores to [(1-self.param), 1] if the score is not bounded.

        Note:
            This replace the temperature scaling process of this confidence scores, e.g., doctor, energy.

        Args:
            scores: The unbounded confidence scores, of shape ``(n, )`` or a list of n ``(H, W, (D))``.
            e1: A small number to prevent unexpected results of division.
            lb: Lower bound of scores to calibrate the confidence scores of shape ``(d, )`` for class-specific or ``(1, )`` for class-agnostic.
                If it is omited as None, the scores are normalized to [0, 1] (lb = 0).
            pred: The prediction derived from the logits, of shape ``(n, )`` or a list of n ``(H, W, (D))``, used for class-specific operation.
        
        Returns:
            normalized_scores: The normalized confidence scores, of shape ``(n, )`` or a list of n ``(H, W, (D))``.
        
        """

        if self.is_training:
            
            self._find_normalize(scores)

        if isinstance(scores, list):
            
            _normalized_scores = []
            for n_case in range(len(scores)):
                _normalized_scores.append( (scores[n_case] - self.min_value) / (self.max_value - self.min_value + e1) )

            if lb is None:
                return _normalized_scores
            
            else:

                normalized_scores = []

                for n_case in range(len(_normalized_scores)):

                    # reshape normalized_scores, pred to vectors.
                    inp_shape = _normalized_scores[n_case].shape

                    # flaten the logit for segmentation.
                    normalized_score = _normalized_scores[n_case].reshape((-1))
                    pred_case = pred[n_case].reshape((-1))

                    # normalized_scores now is of shape ``(n, )``
                    # pred now is of shape ``(n, )``

                    if not self.class_specific:
                        normalized_score = normalized_score * self.param + (1 - self.param)
                    else:
                        # class-wise average confidence
                        for kcls in range(self.num_class):
                            pos_cls = np.where(pred_case == kcls)[0]
                            normalized_score[pos_cls] = normalized_score[pos_cls] * self.param[kcls] + (1 - self.param[kcls])
                    
                    normalized_score = normalized_score.reshape(inp_shape)

                    normalized_scores.append(normalized_score)

                return normalized_scores

        else:

            normalized_scores = (scores - self.min_value) / (self.max_value - self.min_value + e1)

            if lb is None:
                return normalized_scores
            else:
                # normalized_scores now is of shape ``(n, )``
                # pred now is of shape ``(n, )``

                if not self.class_specific:
                    normalized_scores = normalized_scores * self.param + (1 - self.param)
                else:
                    # class-wise average confidence
                    for kcls in range(self.num_class):
                        pos_cls = np.where(pred == kcls)[0]
                        normalized_scores[pos_cls] = normalized_scores[pos_cls] * self.param[kcls] + (1 - self.param[kcls])

                return normalized_scores
    
    def __call__(self, inp: Union[List[Iterable], np.ndarray], midstage: bool = False, gt: List[Iterable] = None) -> float:
        """Calculate the performance using network output.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            midstage: If ``True``, return the first calibrated results.
            gt: The cooresponding annotation of a list of n ``(H, W, (D))`` for segmentation. This is only rquired for segmentation task during optimizaing.
                We will utilize this to determine if there is label in the case. If not, we do not calculate the dsc and utilize for optimizing.
                This is because we do not want to optimize parameter with blank segmentation map.
        
        Returns:
            estim_acc: A float scalar that represents the estimated accuracy for the given input data.
            estim_cls: Estimated class-wise performance of shape ``(d, )``.
                For classification task, this is the class-wise estimated accuracy.
                For segmentation task, this class-wise estimated accuracy for background and DSC estimation for other foreground classes.

        """

        # calibrate the confidence scores
        if self.extend_param:
            score = self.calibrate(inp, midstage)
        else:
            score = self.calibrate(inp)

        # Estimate the performance.
        if self.mode == "classification":
            # for classification tasks, return average confidence
            estim_acc = np.mean(score)
            # class-wise average confidence
            pred = np.argmax(inp, axis = 1)
            estim_acc_allcls = []
            for kcls in range(inp.shape[1]):
                score_kcls = score[np.where(pred == kcls)[0]]
                estim_acc_kcls = np.mean(score_kcls)
                estim_acc_allcls.append(estim_acc_kcls)
            return estim_acc, np.array(estim_acc_allcls)
        elif self.mode == "segmentation":
            # for segmentation tasks, return average confidence for background class (c=0), and return the soft dsc for other foreground classes.
            scores = []
            scores_bg = []
            for n_case in range(len(inp)):
                pred_case = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                pred_flatten = pred_case.flatten() # ``n``
                score_flatten = score[n_case].flatten() # ``n``
                score_flatten_bg = score_flatten[np.where(pred_flatten == 0)[0]]
                scores.append(score_flatten)
                scores_bg.append(score_flatten_bg)

            estim_acc = np.mean( np.concatenate(scores) )
            estim_acc_bg = np.mean( np.concatenate(scores_bg) )
            
            # Note, the calculation of dice score need the probability of non-maximum classes 
            # here we extend score from shape ``(n, H, W, (D))`` to ``(n, d, H, W, (D))``, the other dimension are just filled with zeros
            estim_dsc_list = []
            for n_case in range(len(inp)):
                pred_case = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                pred_flatten = pred_case.flatten() # ``n``
                score_case = score[n_case] # ``(H, W, (D))``
                score_flatten = score[n_case].flatten() # ``n``
                score_filled = np.zeros((score_flatten.shape + (self.num_class,))) # ``(n, d)``
                score_filled[np.arange(score_filled.shape[0]), pred_flatten] = score_flatten # ``(n, d)``
                score_filled = score_filled.T # ``(d, n)``
                score_filled = score_filled.reshape(((self.num_class,) + score_case.shape)) # ``(d, H, W, (D))``
                #
                estim_dsc = SoftDiceLoss(score_filled[np.newaxis, ...], pred_case[np.newaxis, ...])
                #
                if gt != None:
                    gt_case = gt[n_case]
                    # We remove the class DSC if there isn't any in gt
                    for k_cls in range(len(estim_dsc)):
                        if np.sum(gt_case == k_cls) == 0:
                            estim_dsc[k_cls] = 0

                estim_dsc_list.append(estim_dsc)
            m_estim_dsc = np.mean(np.array(estim_dsc_list), axis=0)
            
            return estim_acc, np.concatenate([np.array([estim_acc_bg]), m_estim_dsc[1:]])
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def calibrate(self, scores: Union[List[Iterable], np.ndarray]) -> Union[List[Iterable], np.ndarray]:
        """Calibrate the confidence scores with parameters.

        Different estimation algorithms would adopt different strateiges to calibrate the scores.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data.
        
        """
        
        raise NotImplementedError()


@register("ac-model")
class acModel(Model):
    """MOVAL model with average confidence."""

    def __init__(self, mode: str, num_class: int, confidence_scores: str, class_specific: bool):
        super().__init__(
            estim_algorithm = "ac-model",
            mode = mode,
            num_class = num_class,
            confidence_scores = confidence_scores,
            class_specific = class_specific,
            extend_param = False
        )
    
    def calibrate(self, inp: Union[List[Iterable], np.ndarray]) -> Union[List[Iterable], np.ndarray]:
        """Calibrate the confidence scores for average confidence.
        
        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, )`` or a list of n ``(H, W, (D))``.

        """

        # Calualte the confidence socres.
        score = self.conf(inp) # ``(n,)`` for classification or a list of n ``(H, W, (D))`` for segmentation

        # Normalize the scores, if needed.
        if self.conf.normalization:
            score = self._normalize(score)
        
        return score

@register("ts-model")
class tsModel(Model):
    """MOVAL model with temperature scaling."""

    def __init__(self, mode: str, num_class: int, confidence_scores: str, class_specific: bool):
        super().__init__(
            estim_algorithm = "ts-model",
            mode = mode,
            num_class = num_class,
            confidence_scores = confidence_scores,
            class_specific = class_specific,
            extend_param = False
        )
    
    def calibrate(self, inp: Union[List[Iterable], np.ndarray]) -> Union[List[Iterable], np.ndarray]:
        """Calibrate the confidence scores with temperature scaling.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, )`` or a list of n ``(H, W, (D))``.
        
        """

        # Calualte the confidence socres.
        score = self.conf(inp, self.param) # ``(n,)`` for classification or a list of n ``(H, W, (D))`` for segmentation

        # Normalize the scores, if needed.
        if self.conf.normalization:
            if self.mode == "classification":
                pred = np.argmax(inp, axis = 1)
                score = self._normalize(score, lb = self.param, pred = pred)
            elif self.mode == "segmentation":
                preds = []
                for n_case in range(len(score)):
                    pred_case = np.argmax(inp[n_case], axis = 0)
                    preds.append(pred_case)
                score = self._normalize(score, lb = self.param, pred = preds)
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")
        
        return score

@register("doc-model")
class docModel(Model):
    """MOVAL model with difference of confidence."""

    def __init__(self, mode: str, num_class: int, confidence_scores: str, class_specific: bool):
        super().__init__(
            estim_algorithm = "doc-model",
            mode = mode,
            num_class = num_class,
            confidence_scores = confidence_scores,
            class_specific = class_specific,
            extend_param = False
        )
    
    def calibrate(self, inp: Union[List[Iterable], np.ndarray]) -> Union[List[Iterable], np.ndarray]:
        """Calibrate the confidence scores based on difference of confidence.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, )`` or a list of n ``(H, W, (D))``.
        
        Note:
            The implementation of class-specific DoC for segmentation is different from what we did in our MICCAI work.
            In MICCAI, we calculate the difference as the difference of soft dsc. Here, we modify the confidence score to match dsc.
            Current version should be more compatable with other techiques.
        
        """

        # Calualte the confidence socres.
        score = self.conf(inp) # ``(n,)`` for classification or a list of n ``(H, W, (D))`` for segmentation

        # Normalize the scores, if needed.
        if self.conf.normalization:
            score = self._normalize(score)

        if not self.class_specific:
            if self.mode == "classification":
                return score - (1 - self.param)
            elif self.mode == "segmentation":
                score_post = []
                for n_case in range(len(score)):
                    score_case = score[n_case] - (1 - self.param)
                    score_post.append(score_case)
                return score_post
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")
        else:
            # class-wise average confidence
            if self.mode == "classification":
                pred = np.argmax(inp, axis = 1)
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(pred == kcls)[0]
                    score[pos_cls] = score[pos_cls] - (1 - self.param[kcls])
                return score
            elif self.mode == "segmentation":
                score_post = []
                for n_case in range(len(score)):
                    score_case = score[n_case] # ``(H, W, (D))``
                    pred_case = np.argmax(inp[n_case], axis = 0)
                    score_shape = score_case.shape
                    score_flatten = score_case.flatten()
                    pred_flatten = pred_case.flatten()
                    #
                    for kcls in range(self.num_class):
                        pos_flatten_cls = np.where(pred_flatten == kcls)[0]
                        score_flatten[pos_flatten_cls] = score_flatten[pos_flatten_cls] - (1 - self.param[kcls])
                        score_reshape = score_flatten.reshape(score_shape) # ``(H, W, (D))``
                    score_post.append(score_reshape)
                return score_post
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")

@register("atc-model")
class atcModel(Model):
    """MOVAL model with average thresholded confidence."""

    def __init__(self, mode: str, num_class: int, confidence_scores: str, class_specific: bool):
        super().__init__(
            estim_algorithm = "atc-model",
            mode = mode,
            num_class = num_class,
            confidence_scores = confidence_scores,
            class_specific = class_specific,
            extend_param = False
        )

    def calibrate(self, inp: Union[List[Iterable], np.ndarray]) -> Union[List[Iterable], np.ndarray]:
        """Calibrate the confidence scores based on average thresholded confidence.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, )`` or a list of n ``(H, W, (D))``.
        
        """

        # Calualte the confidence socres.
        _score = self.conf(inp) # ``(n,)`` for classification or a list of n ``(H, W, (D))`` for segmentation

        # Normalize the scores, if needed.
        if self.conf.normalization:
            _score = self._normalize(_score)

        if not self.class_specific:
            if self.mode == "classification":
                score = np.zeros(inp.shape[0])
                score[_score > self.param] = 1
                return score
            elif self.mode == "segmentation":
                score_post = []
                for n_case in range(len(_score)):
                    score = np.zeros(_score[n_case].shape)
                    score[_score[n_case] > self.param] = 1
                    score_post.append(score)
                return score_post
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")
        else:
            # class-wise average confidence
            if self.mode == "classification":
                score = np.zeros(inp.shape[0])
                pred = np.argmax(inp, axis = 1)
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(pred == kcls)[0]
                    score_pos = score[pos_cls]
                    score_pos[_score[pos_cls] > self.param[kcls]] = 1
                    score[pos_cls] = score_pos
                return score
            elif self.mode == "segmentation":
                score_post = []
                for n_case in range(len(_score)):
                    _score_case = _score[n_case] # ``(H, W, (D))``
                    score_case = np.zeros(_score[n_case].shape)
                    pred_case = np.argmax(inp[n_case], axis = 0)
                    score_shape = score_case.shape
                    pred_flatten = pred_case.flatten()
                    score_flatten = score_case.flatten()
                    _score_flatten = _score_case.flatten()
                    #
                    for kcls in range(self.num_class):
                        pos_flatten_cls = np.where(pred_flatten == kcls)[0]
                        score_pos = score_flatten[pos_flatten_cls]
                        score_pos[_score_flatten[pos_flatten_cls] > self.param[kcls]] = 1
                        score_flatten[pos_flatten_cls] = score_pos
                        score_reshape = score_flatten.reshape(score_shape) # ``(H, W, (D))``
                    score_post.append(score_reshape)
                return score_post
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")

@register("ts-atc-model")
class tsatcModel(Model):
    """MOVAL model with average thresholded confidence after temperature scaling."""

    def __init__(self, mode: str, num_class: int, confidence_scores: str, class_specific: bool):
        super().__init__(
            estim_algorithm = "ts-atc-model",
            mode = mode,
            num_class = num_class,
            confidence_scores = confidence_scores,
            class_specific = class_specific,
            extend_param = True
        )
    
    def calibrate(self, inp: Union[List[Iterable], np.ndarray], midstage: bool = False) -> Union[List[Iterable], np.ndarray]:
        """Calibrate the confidence scores based on average thresholded confidence after temperature scaling.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            midstage: If ``True``, return the first calibrated results.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, )`` or a list of n ``(H, W, (D))``.
        
        """

        # Calualte the confidence socres.
        _score = self.conf(inp, self.param) # ``(n,)`` for classification or a list of n ``(H, W, (D))`` for segmentation
        
        # Normalize the scores, if needed.
        if self.conf.normalization:
            if self.mode == "classification":
                pred = np.argmax(inp, axis = 1)
                _score = self._normalize(_score, lb = self.param, pred = pred)
            elif self.mode == "segmentation":
                preds = []
                for n_case in range(len(_score)):
                    pred_case = np.argmax(inp[n_case], axis = 0)
                    preds.append(pred_case)
                _score = self._normalize(_score, lb = self.param, pred = preds)
            else:
                raise ValueError(f"Unknown mode '{self.mode}'") 

        if midstage:
            return _score

        if not self.class_specific:
            if self.mode == "classification":
                score = np.zeros(inp.shape[0])
                score[_score > self.param_ext] = 1
                return score
            elif self.mode == "segmentation":
                score_post = []
                for n_case in range(len(_score)):
                    score = np.zeros(_score[n_case].shape)
                    score[_score[n_case] > self.param_ext] = 1
                    score_post.append(score)
                return score_post
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")
        else:
            # class-wise average confidence
            if self.mode == "classification":
                score = np.zeros(inp.shape[0])
                pred = np.argmax(inp, axis = 1)
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(pred == kcls)[0]
                    score_pos = score[pos_cls]
                    score_pos[_score[pos_cls] > self.param_ext[kcls]] = 1
                    score[pos_cls] = score_pos
                return score
            elif self.mode == "segmentation":
                score_post = []
                for n_case in range(len(_score)):
                    _score_case = _score[n_case] # ``(H, W, (D))``
                    score_case = np.zeros(_score[n_case].shape)
                    pred_case = np.argmax(inp[n_case], axis = 0)
                    score_shape = score_case.shape
                    pred_flatten = pred_case.flatten()
                    score_flatten = score_case.flatten()
                    _score_flatten = _score_case.flatten()
                    #
                    for kcls in range(self.num_class):
                        pos_flatten_cls = np.where(pred_flatten == kcls)[0]
                        score_pos = score_flatten[pos_flatten_cls]
                        score_pos[_score_flatten[pos_flatten_cls] > self.param_ext[kcls]] = 1
                        score_flatten[pos_flatten_cls] = score_pos
                        score_reshape = score_flatten.reshape(score_shape) # ``(H, W, (D))``
                    score_post.append(score_reshape)
                return score_post
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")

        return score