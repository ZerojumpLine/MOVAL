# Adapted from
# https://github.com/melanibe/failure_detection_benchmark

import abc
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union

from moval.models import register
from moval.models.utils import cal_energy, cal_mcp, cal_entropy, cal_doctor
import numpy as np

class Confidence(abc.ABC):
    """Base model for confidence calculation.

    Confidence scores can be computed by calling the ``__call__()`` method. This class should not 
    be directly instaniated, and instead used as the base class for confidence scores calculation.

    Atrributes:
        score_type (str):
            The name of confidence scores.
        negative_score (bool):
            A boolean that indicates if original confidence score is negative correlated to the classification correctness.
        normalization (bool):
            A boolean that indicates if original confidence score needs normalization before usage.
        cal_func (:py:func):
            The function to calculate the corresponding confidence scores.

    """

    def __init__(self, score_type: str, negative_score: bool, normalization: bool):
        self.score_type = score_type
        self.negative_score = negative_score
        self.normalization = normalization
        
        if self.score_type == "max_class_probability-conf":
            self.cal_func = cal_mcp
        elif self.score_type == "energy-conf":
            self.cal_func = cal_energy
        elif self.score_type == "entropy-conf":
            self.cal_func = cal_entropy
        elif self.score_type == "doctor-conf":
            self.cal_func = cal_doctor
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
    
    def reshape_T(self, T: Union[type(None), np.ndarray], num_class: int) -> np.ndarray:
        """Reshape T from arbitrary shape to ``(d, )``.

        Args:
            T: Temperature to calibrate the confidence scores of type None, shape ``(d, )`` or a numpy float.
            num_class: the number of class for the logit.
        
        Returns:
            T: Reshaped T of shape ``(d, )``.
        
        """
        if T is None:
            T = np.ones(num_class)
        if len(T) == 1:
            T = T.repeat(num_class)
        
        return T
    
    def reshape_inp(self, inp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape the input from arbitrary shape to ``(n, d)``.

        Args:
            inp: The network output (logits) of shape ``(n, d, (H), (W), (D))``.
        
        Returns:
            inp: Reshaped network output of shape ``(n, d)``.
            inp_shape: The shape of original network output.
        
        """
        
        inp_shape = inp.shape

        if len(inp_shape) > 2:
            # flaten the logit for segmentation.
            # first tranpose the feature to the last dimension
            axes = [0] + list(range(2, len(inp_shape))) + [1]
            inp = np.transpose(inp, axes)
            # then flatten all
            inp = inp.reshape((-1,) + inp.shape[-1:])
        
        return inp, inp_shape
    
    def __call__(self, inp: np.ndarray, T: np.ndarray = None) -> np.ndarray:
        """Calculate softmax probability and return the maximum.

        Args:
            inp: The network output (logits) of shape ``(n, d, (H), (W), (D))``.
            T: Tempratature to calibrate the confidence scores of shape ``(d, )`` for class-specific or ``(1, )`` for class-agnostic.
                If it is omited as None, the scores are not calibrated (T = 1).
        
        Return:
            conf_score: The calculated confidence scores, of shape ``(n, (H), (W), (D))``.
        
        """

        num_class = inp.shape[1]
        T = self.reshape_T(T, num_class=num_class)
        # now T is of shape ``(d, )``
        
        inp, inp_shape = self.reshape_inp(inp)
        # now inp is of shape ``(n, d)``

        pred = np.argmax(inp, axis=1)
        conf_score = np.zeros(inp.shape[0]) # ``(n, )``
        for kcls in range(num_class):
            pred_cls_position,  = np.where(pred == kcls)
            conf_score[pred_cls_position] = self.cal_func(inp[pred_cls_position, :], T[kcls])

        if self.negative_score:
            conf_score = 1 - conf_score

        if len(inp_shape) > 2:
            # transfer back to the original dimension
            conf_score = conf_score.reshape(inp_shape[:1] + inp_shape[2:])

        return conf_score

@register("max_class_probability-conf")
class maxclassprobabilityConfidence(Confidence):
    """Confidence scores based on max class probability.
    
    Note:
        The maximum class probability ([0, 1]) is high is the model is confident.
    
    """

    def __init__(self):
        super().__init__(
            score_type = "max_class_probability-conf",
            negative_score = False,
            normalization = False
        )
    
    
# energy scores
@register("energy-conf")
class energyConfidence(Confidence):
    """Confidence scores based on energy.
    
    Note:
        Energy is low is the model is confident, we should calculate negative energy.
        Original engergy is unbounded, we should normalize by finding the max/min value later.

    """

    def __init__(self):
        super().__init__(
            score_type = "energy-conf",
            negative_score = True,
            normalization = True
        )

# entropy
@register("entropy-conf")
class energyConfidence(Confidence):
    """Confidence scores based on entropy.
    
    Note:
        Entropy is low is the model is confident, we should calculate negative entropy.
        We should also choose the base as class number to ensure that the largest entropy is 1.

    """

    def __init__(self):
       super().__init__(
            score_type = "entropy-conf",
            negative_score = True,
            normalization = False
        )
    

# doctor scores
@register("doctor-conf")
class doctorConfidence(Confidence):
    """MOVAL model with average confidence."""

    def __init__(self):
        super().__init__(
            score_type = "doctor-conf",
            negative_score = True,
            normalization = True
        )
    