import abc
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

        if self.conf.normalization:
            self.max_value = 0.
            self.min_value = 0.
        
        self.is_training = False
        
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

    def _find_normalize(self, scores: np.ndarray):
        """Find the normalization parameter for the confidence scores.

        Args:
            scores: The unbounded confidence scores, of shape ``(n, (H), (W), (D))``.

        Notes:
            The intensity value, max and min value, will be saved.
        
        """

        self.max_value = np.max(scores)
        self.min_value = np.min(scores)

    def _normalize(self, scores: np.ndarray, e1: float = 1e-6) -> np.ndarray:
        """Normalize the confidence scores to [0,1] if the score is not bounded.

        Args:
            scores: The unbounded confidence scores, of shape ``(n, (H), (W), (D))``.
            e1: A small number to prevent unexpected results of division.
        
        Returns:
            normalized_scores: The normalized confidence scores, of shape ``(n, (H), (W), (D))``.
        
        """

        if self.is_training:
            self._find_normalize(scores)

        if self.conf.normalization:
            normalized_scores = (scores - self.min_value) / (self.max_value - self.min_value + e1)
            return normalized_scores
        
        else:
            return scores
    
    def __call__(self, inp: np.ndarray) -> float:
        """Calculate the performance using network output.

        Args:
            inp: The network output (logits) of shape ``(n, d, (H), (W), (D))``.
        
        Returns:
            estim_acc: A float scalar that represents the estimated accuracy for the given input data.
            estim_cls: Estimated class-wise performance of shape ``(d, )``.
                For classification task, this is the class-wise estimated accuracy.
                For segmentation task, this class-wise estimated accuracy for background and DSC estimation for other foreground classes.

        """

        # calibrate the confidence scores
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
            pred = np.argmax(inp, axis = 1)
            pred_flatten = pred.flatten()
            score_flatten = score.flatten()
            score_flatten_bg = score_flatten[np.where(pred_flatten == 0)[0]]
            estim_acc = np.mean(score_flatten)
            estim_acc_bg = np.mean(score_flatten_bg)
            # Note, the calculation of dice score need the probability of non-maximum classes 
            # here we extend score from shape ``(n, H, W, (D))`` to ``(n, d, H, W, (D))``, the other dimension are just filled with zeros
            score_filled = np.zeros((pred_flatten.shape[-1], inp.shape[1])) # ``(n * H * W * (D), d)``
            score_filled[np.arange(score_filled.shape[0]), pred_flatten] = score_flatten
            score_filled = score_filled.reshape(inp.shape[:1] + inp.shape[2:] + inp.shape[1:2]) # ``(n, H, W, (D), d)``
            axes = [0] + [len(score_filled.shape)-1] + list(range(1, len(score_filled.shape)-1))
            score_filled = np.transpose(score_filled, axes) # ``(n, d, H, W, (D))``
            #
            estim_dsc = SoftDiceLoss(score_filled, pred)
            return estim_acc, np.concatenate([np.array([estim_acc_bg]), estim_dsc[1:]])
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
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
    
    def calibrate(self, inp: np.ndarray) -> np.ndarray:
        """Calibrate the confidence scores for average confidence.
        
        Args:
            inp: The network output (logits) of shape ``(n, d, (H), (W), (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, (H), (W), (D))``.

        """

        # Calualte the confidence socres.
        score = self.conf(inp) # `(n,)` for classification or `(n, H, W, (D))` for segmentation

        # Normalize the scores, if needed.
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
    
    def calibrate(self, inp: np.ndarray) -> np.ndarray:
        """Calibrate the confidence scores with temperature scaling.

        Args:
            inp: The network output (logits) of shape ``(n, d, (H), (W), (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, (H), (W), (D))``.
        
        """

        # Calualte the confidence socres.
        score = self.conf(inp, self.param) # `(n,)` for classification or `(n, H, W, (D))` for segmentation

        # Normalize the scores, if needed.
        score = self._normalize(score)
        
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
    
    def calibrate(self, inp: np.ndarray) -> np.ndarray:
        """Calibrate the confidence scores based on difference of confidence.

        Args:
            inp: The network output (logits) of shape ``(n, d, (H), (W), (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, (H), (W), (D))``.
        
        Note:
            The implementation of class-specific DoC for segmentation is different from what we did in our MICCAI work.
            In MICCAI, we calculate the difference as the difference of soft dsc. Here, we modify the confidence score to match dsc.
            Current version should be more compatable with other techiques.
        
        """

        # Calualte the confidence socres.
        score = self.conf(inp) # `(n,)` for classification or `(n, H, W, (D))` for segmentation

        if not self.class_specific:
            score = score - (1 - self.param)
        else:
            # class-wise average confidence
            pred = np.argmax(inp, axis = 1)
            if self.mode == "classification":
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(pred == kcls)[0]
                    score[pos_cls] = score[pos_cls] - (1 - self.param[kcls])
            elif self.mode == "segmentation":
                pred_flatten = pred.flatten()
                score_flatten = score.flatten()
                #
                for kcls in range(inp.shape[1]):
                    pos_flatten_cls = np.where(pred_flatten == kcls)[0]
                    score_flatten[pos_flatten_cls] = score_flatten[pos_flatten_cls] - (1 - self.param[kcls])
                    score = score_flatten.reshape(inp.shape[:1] + inp.shape[2:]) # ``(n, H, W, (D))``
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")
        
        # Normalize the scores, if needed.
        score = self._normalize(score)
        
        return score

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

    def calibrate(self, inp: np.ndarray) -> np.ndarray:
        """Calibrate the confidence scores based on average thresholded confidence.

        Args:
            inp: The network output (logits) of shape ``(n, d, (H), (W), (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, (H), (W), (D))``.
        
        """

        # Calualte the confidence socres.
        _score = self.conf(inp) # `(n,)` for classification or `(n, H, W, (D))` for segmentation

        score = np.zeros((inp.shape[:1] + inp.shape[2:]))
        if not self.class_specific:
            score[_score > self.param] = 1
        else:
            # class-wise average confidence
            pred = np.argmax(inp, axis = 1)
            if self.mode == "classification":
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(pred == kcls)[0]
                    score[pos_cls][_score[pos_cls] > self.param[kcls]] = 1
            elif self.mode == "segmentation":
                pred_flatten = pred.flatten()
                score_flatten = score.flatten()
                _score_flatten = _score.flatten()
                #
                for kcls in range(inp.shape[1]):
                    pos_flatten_cls = np.where(pred_flatten == kcls)[0]
                    score_flatten[pos_flatten_cls][_score_flatten[pos_flatten_cls] > self.param[kcls]] = 1
                    score = score_flatten.reshape(inp.shape[:1] + inp.shape[2:]) # ``(n, H, W, (D))``
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")
        
        score = self._normalize(score)
        
        return score

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
    
    def calibrate(self, inp: np.ndarray) -> np.ndarray:
        """Calibrate the confidence scores based on average thresholded confidence after temperature scaling.

        Args:
            inp: The network output (logits) of shape ``(n, d, (H), (W), (D))``.

        Returns:
            calibrated_scores: The calibrated scores which would match the accuracy/DSC on validation data, of shape ``(n, (H), (W), (D))``.
        
        """

        # Calualte the confidence socres.
        _score = self.conf(inp, self.param) # `(n,)` for classification or `(n, H, W, (D))` for segmentation

        score = np.zeros((inp.shape[:1] + inp.shape[2:]))
        if not self.class_specific:
            score[_score > self.param_ext] = 1
        else:            
            # class-wise average confidence
            pred = np.argmax(inp, axis = 1)
            if self.mode == "classification":
                for kcls in range(inp.shape[1]):
                    pos_cls = np.where(pred == kcls)[0]
                    score[pos_cls][_score[pos_cls] > self.param[kcls]] = 1
            elif self.mode == "segmentation":
                pred_flatten = pred.flatten()
                score_flatten = score.flatten()
                _score_flatten = _score.flatten()
                #
                for kcls in range(inp.shape[1]):
                    pos_flatten_cls = np.where(pred_flatten == kcls)[0]
                    score_flatten[pos_flatten_cls][_score_flatten[pos_flatten_cls] > self.param[kcls]] = 1
                    score = score_flatten.reshape(inp.shape[:1] + inp.shape[2:]) # ``(n, H, W, (D))``
            else:
                raise ValueError(f"Unknown mode '{self.mode}'")
        
        # Normalize the scores, if needed.
        score = self._normalize(score)

        return score