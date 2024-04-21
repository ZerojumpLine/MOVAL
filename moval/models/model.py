import abc
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
from moval.models import register
import moval.models
from moval.models.utils import SoftDiceLoss, SoftSensitivity, SoftPrecision, SoftAUC, cal_softmax
from moval.models.solver_temperature import solve_T

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
        self.normalized = False
        
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
            The intensity value, max and min value, will be saved. This should be only done with the T = 1 (inital states).
        
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

    def _normalize(self, scores: Union[List[Iterable], np.ndarray], e1: float = 1e-6) -> Union[List[Iterable], np.ndarray]:
        """Normalize the confidence scores to [0, 1] if the score is not bounded.

        Note:
            This replace the temperature scaling process of this confidence scores, e.g., doctor, energy.

        Args:
            scores: The unbounded confidence scores, of shape ``(n, )`` or a list of n ``(H, W, (D))``.
            e1: A small number to prevent unexpected results of division.
        
        Returns:
            normalized_scores: The normalized confidence scores, of shape ``(n, )`` or a list of n ``(H, W, (D))``.
        
        """

        if self.is_training and not self.normalized:
            
            self._find_normalize(scores)
            self.normalized = True

        if isinstance(scores, list):
            
            normalized_scores = []
            for n_case in range(len(scores)):
                normalized_score =  (scores[n_case] - self.min_value) / (self.max_value - self.min_value + e1)
                # normalized_score has shape ``(H, W, (D))``
                # restrict the scores to the range of [0, 1]
                normalized_score = np.clip(normalized_score, 0, 1)
                normalized_scores.append(normalized_score)
        else:

            normalized_scores = (scores - self.min_value) / (self.max_value - self.min_value + e1)
            # normalized_scores has shape ``(n, )``
            # restrict the scores to the range of [0, 1]
            normalized_scores = np.clip(normalized_scores, 0, 1)
                
        return normalized_scores
    
    def estimate_accuracy(self, inp: Union[List[Iterable], np.ndarray], midstage: bool = False, gt_guide: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """Estimate the accuracy using network output.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            midstage: If ``True``, return the first calibrated results.
            gt_guide: The cooresponding annotation guide of shape ``(n, d)`` for segmentation. This is only rquired for segmentation task during optimizaing.
                We will utilize this to determine if there is label in the case. If not, we do not calculate the dsc and utilize for optimizing.
                This is because we do not want to optimize parameter with blank segmentation map. 
                This should be bool, if ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim_acc: A float scalar that represents the estimated accuracy for the given input data.
            estim_cls: Estimated class-wise accuracy of shape ``(d, )``.

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
                if len(score_kcls) > 0:
                    estim_acc_kcls = np.mean(score_kcls)
                else:
                    # does not pred any
                    estim_acc_kcls = 1e-5
                estim_acc_allcls.append(estim_acc_kcls)
            return estim_acc, np.array(estim_acc_allcls)
        elif self.mode == "segmentation":
            # for segmentation tasks
            scores = []
            for n_case in range(len(inp)):
                score_flatten = score[n_case].flatten() # ``n``
                scores.append(score_flatten)
            estim_acc = np.mean( np.concatenate(scores) )

            estim_acc_allcls = []
            for kcls in range(inp[0].shape[0]):
                scores_kcls = []
                for n_case in range(len(inp)):
                    pred_case = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                    pred_flatten = pred_case.flatten() # ``n``
                    score_flatten = score[n_case].flatten() # ``n``
                    score_flatten_kcls = score_flatten[np.where(pred_flatten == kcls)[0]]
                    scores_kcls.append(score_flatten_kcls)
                estim_acc_allcls.append(np.mean( np.concatenate(scores_kcls) ))
            
            return estim_acc, np.array(estim_acc_allcls)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def estimate_sensitivity(self, inp: Union[List[Iterable], np.ndarray], probability: Union[List[Iterable], np.ndarray], gt_guide: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """Esimate the sensitivity using network output.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            probability: The calibrated probability of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            gt_guide: The cooresponding annotation guide of shape ``(n, d)`` for segmentation. This is only rquired for segmentation task during optimizaing.
                We will utilize this to determine if there is label in the case. If not, we do not calculate the dsc and utilize for optimizing.
                This is because we do not want to optimize parameter with blank segmentation map. 
                This should be bool, if ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Note:
            The user may wonder why we need inp here in this function. It is because we utilize inp to determine the prediction results, which are not always feasible by probability.

        Returns:
            estim_sensitivity: Estimated class-wise sensitivity of shape ``(d, )``.

        """

        # Estimate the sensitivity.
        if self.mode == "classification":
            # probability is of shape ``(n, d)``
            estim_sensitivity = SoftSensitivity(probability, np.argmax(inp, axis = 1))
            return estim_sensitivity
        elif self.mode == "segmentation":
            # probability is a list of n ``(d, H, W, (D))``.
            estim_sensitivity_list = []
            for n_case in range(len(inp)):
                pred_case = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                score_filled = probability[n_case] # ``(d, H, W, (D))``
                #
                estim_sensitivity = SoftSensitivity(score_filled[np.newaxis, ...], pred_case[np.newaxis, ...])
                #
                if isinstance(gt_guide, np.ndarray):
                    gt_case = gt_guide[n_case]
                    # We remove the class DSC if there isn't any in gt
                    for kcls in range(len(estim_sensitivity)):
                        if gt_case[kcls] is False:
                            estim_sensitivity[kcls] = -1

                estim_sensitivity_list.append(estim_sensitivity)
            
            # aggregate the results
            estim_sensitivity_list = np.array(estim_sensitivity_list) # ``(n, d)``
            m_estim_sensitivity = []
            for kcls in range(len(estim_sensitivity)):
                m_estim_sensitivity.append(estim_sensitivity_list[:, kcls][estim_sensitivity_list[:,kcls] >= 0].mean())
            
            return np.array(m_estim_sensitivity)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
    
    def estimate_precision(self, probability: Union[List[Iterable], np.ndarray], gt_guide: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """Esimate the precision using network output.

        Args:
            probability: The calibrated probability of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            gt_guide: The cooresponding annotation guide of shape ``(n, d)`` for segmentation. This is only rquired for segmentation task during optimizaing.
                We will utilize this to determine if there is label in the case. If not, we do not calculate the dsc and utilize for optimizing.
                This is because we do not want to optimize parameter with blank segmentation map. 
                This should be bool, if ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim_precision: Estimated class-wise precision of shape ``(d, )``.

        """

        if self.estim_algorithm == "atc-model" or self.estim_algorithm == "ts-atc-model":
            raise ValueError(f"estimation algorithm '{self.estim_algorithm}' does not support the estimation of precision (because FP is always 0)")

        # Estimate the sensitivity.
        if self.mode == "classification":
            # probability is of shape ``(n, d)``
            estim_precision = SoftPrecision(probability, np.argmax(probability, axis = 1))
            return estim_precision
        elif self.mode == "segmentation":
            # probability is a list of n ``(d, H, W, (D))``.
            estim_precision_list = []
            for n_case in range(len(probability)):
                pred_case = np.argmax(probability[n_case], axis = 0) # ``(H, W, (D))``
                score_filled = probability[n_case] # ``(d, H, W, (D))``
                #
                estim_precision = SoftPrecision(score_filled[np.newaxis, ...], pred_case[np.newaxis, ...])
                #
                if isinstance(gt_guide, np.ndarray):
                    gt_case = gt_guide[n_case]
                    # We remove the class DSC if there isn't any in gt
                    for kcls in range(len(estim_precision)):
                        if gt_case[kcls] is False:
                            estim_precision[kcls] = -1

                estim_precision_list.append(estim_precision)
            
            # aggregate the results
            estim_precision_list = np.array(estim_precision_list) # ``(n, d)``
            m_estim_precision = []
            for kcls in range(len(estim_precision)):
                m_estim_precision.append(estim_precision_list[:, kcls][estim_precision_list[:,kcls] >= 0].mean())
            
            return np.array(m_estim_precision)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
    
    def estimate_f1score(self, inp: Union[List[Iterable], np.ndarray], probability: Union[List[Iterable], np.ndarray], gt_guide: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """Esimate the F1score using network output.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            probability: The calibrated probability of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            gt_guide: The cooresponding annotation guide of shape ``(n, d)`` for segmentation. This is only rquired for segmentation task during optimizaing.
                We will utilize this to determine if there is label in the case. If not, we do not calculate the dsc and utilize for optimizing.
                This is because we do not want to optimize parameter with blank segmentation map. 
                This should be bool, if ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim_F1score: Estimated class-wise F1score of shape ``(d, )``.

        """

        # Estimate the sensitivity.
        if self.mode == "classification":
            # probability is of shape ``(n, d)``
            estim_F1score = SoftDiceLoss(probability, np.argmax(inp, axis = 1))
            return estim_F1score
        elif self.mode == "segmentation":
            # probability is a list of n ``(d, H, W, (D))``.
            estim_dsc_list = []
            for n_case in range(len(inp)):
                pred_case = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                score_filled = probability[n_case] # ``(d, H, W, (D))``
                #
                estim_dsc = SoftDiceLoss(score_filled[np.newaxis, ...], pred_case[np.newaxis, ...])
                #
                if isinstance(gt_guide, np.ndarray):
                    gt_case = gt_guide[n_case]
                    # We remove the class DSC if there isn't any in gt
                    for kcls in range(len(estim_dsc)):
                        if gt_case[kcls] is False:
                            estim_dsc[kcls] = -1

                estim_dsc_list.append(estim_dsc)
            
            # aggregate the results
            estim_dsc_list = np.array(estim_dsc_list) # ``(n, d)``
            m_estim_dsc = []
            for kcls in range(len(estim_dsc)):
                m_estim_dsc.append(estim_dsc_list[:, kcls][estim_dsc_list[:,kcls] >= 0].mean())
            
            return np.array(m_estim_dsc)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
    
    def estimate_auc(self, 
                     probability: Union[List[Iterable], np.ndarray], 
                     gt_guide: np.ndarray = None, 
                     sel_cls: int = None) -> Tuple[float, np.ndarray]:
        """Esimate the AUC using network output.

        Args:
            probability: The calibrated probability of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            gt_guide: The cooresponding annotation guide of shape ``(n, d)`` for segmentation. This is only rquired for segmentation task during optimizaing.
                We will utilize this to determine if there is label in the case. If not, we do not calculate the dsc and utilize for optimizing.
                This is because we do not want to optimize parameter with blank segmentation map. 
                This should be bool, if ``False``, it means that there isn't any manuel label of class d in this sample.
            sel_cls: The selected class for calculation. If it is None, return all classes.
        
        Returns:
            estim_AUC: Estimated class-wise AUC of shape ``(d, )``, or ``(1, )`` if sel_cls is givien.

        """

        if self.estim_algorithm == "atc-model" or self.estim_algorithm == "ts-atc-model":
            raise ValueError(f"estimation algorithm '{self.estim_algorithm}' does not support the estimation of AUC (because FP is always 0)")

        # Estimate the sensitivity.
        if self.mode == "classification":
            # probability is of shape ``(n, d)``
            estim_AUC, _, _ = SoftAUC(probability, sel_cls = sel_cls)
            return estim_AUC
        elif self.mode == "segmentation":
            # probability is a list of n ``(d, H, W, (D))``.
            estim_AUC_list = []
            for n_case in range(len(probability)):
                score_filled = probability[n_case] # ``(d, H, W, (D))``
                #
                estim_AUC, _, _ = SoftAUC(score_filled[np.newaxis, ...], sel_cls = sel_cls)
                #
                if isinstance(gt_guide, np.ndarray):
                    gt_case = gt_guide[n_case]
                    # We remove the class DSC if there isn't any in gt
                    for kcls in range(len(estim_AUC)):
                        if gt_case[kcls] is False:
                            estim_AUC[kcls] = -1

                estim_AUC_list.append(estim_AUC)
            
            # aggregate the results
            estim_AUC_list = np.array(estim_AUC_list) # ``(n, d)``
            m_estim_auc = []
            for kcls in range(len(estim_AUC)):
                m_estim_auc.append(estim_AUC_list[:, kcls][estim_AUC_list[:,kcls] >= 0].mean())
            
            return np.array(m_estim_auc)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def calibrate(self, inp: Union[List[Iterable], np.ndarray]) -> Union[List[Iterable], np.ndarray]:
        """Calibrate the confidence scores with parameters.

        Different estimation algorithms would adopt different strateiges to calibrate the scores.
        
        """
        
        raise NotImplementedError()
    
    def calculate_probability(self, inp: Union[List[Iterable], np.ndarray], midstage: bool = False, appr: bool = False, full: bool = False) -> Union[List[Iterable], np.ndarray]:
        """Calculate the calibrated probability with parameters.
        For classificaiton tasks, we choose to estimate the pseudo-temperature, for segmentation tasks, we simplify it with 1-socre.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            midstage: If ``True``, return the first calibrated results.
            appr: If ``True``, utilize the approximation version.
            full: If ``False``, we just put the score in and fill the other domains with 0.

        Returns:
            calibrated_probability: The calibrated probability which would match the accuracy/DSC on validation data, of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
        
        """

        if self.mode == "segmentation" or appr:
            return self.calculate_probability_appr(inp, midstage, full)
        elif self.mode == "classification":
            return self.calculate_probability_temperature(inp, midstage, full)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def calculate_probability_temperature(self, inp: Union[List[Iterable], np.ndarray], midstage: bool = False, full : bool = False) -> Union[List[Iterable], np.ndarray]:
        """Calculate the calibrated probability with parameters, based on temperature scaling.
        The challenge is the calculation of non-maximum probability. 
        To achieve this, we calculate the pseudo-temperature for each samples such that the confidence score match the max probability after the temperature scaling process.
        Then, we utilize the pseudo-temperature to scale the probabilities of other classes.

        Note:
            The calibrated probability should have the same dimension with network outputs.
            The calculation is slow as we need to go through every samples.
            Also, it might be not accurate when calculating estimation algorithms other than MCP, as they are in range [0, 1] instead of [1/n, 1].
        
        Update:
            The function is barely used now, as it is too slow!

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            midstage: If ``True``, return the first calibrated results.
            full: If ``False``, we just put the score in and fill the other domains with 0.

        Returns:
            calibrated_probability: The calibrated probability which would match the accuracy/DSC on validation data, of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
        
        """

        if self.extend_param:
            score = self.calibrate(inp, midstage)
        else:
            score = self.calibrate(inp)
            
        if self.mode == "classification":
            # extend from ``(n, )`` to ``(n, d)``
            if not full or self.estim_algorithm == "atc-model" or (self.estim_algorithm == "ts-atc-model" and midstage == False):
                # ATC cannot get the calibrated probability of all classes, we just fill the other dimension with zeros
                pred_flatten = np.argmax(inp, axis = 1)
                probability = np.zeros((score.shape + (self.num_class,))) # ``(n, d)``
                probability[np.arange(probability.shape[0]), pred_flatten] = score # ``(n, d)``
            else:
                T = solve_T(inp, score) # return T of shape ``(n, )``

                probability = np.zeros((score.shape + (self.num_class,))) # ``(n, d)``
                for ksample in range(len(score)):
                    probability[ksample, :] = cal_softmax(inp[ksample: ksample + 1], T[ksample]) # ``(1, d)``

        elif self.mode == "segmentation":
            # extend from a list of n ``(H, W, (D))`` to a list of n ``(d, H, W, (D))``
            if not full or self.estim_algorithm == "atc-model" or (self.estim_algorithm == "ts-atc-model" and midstage == False):
                # ATC cannot get the calibrated probability of all classes, we just fill the other dimension with zeros
                probability = []
                for n_case in range(len(inp)):
                    pred_case = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                    #
                    pred_flatten = pred_case.flatten() # ``n``
                    score_case = score[n_case] # ``(H, W, (D))``
                    score_flatten = score[n_case].flatten() # ``n``
                    probability_case = np.zeros((score_flatten.shape + (self.num_class,))) # ``(n, d)``
                    probability_case[np.arange(probability_case.shape[0]), pred_flatten] = score_flatten # ``(n, d)``
                    probability_case = probability_case.T # ``(d, n)``
                    probability_case = probability_case.reshape(((self.num_class,) + score_case.shape)) # ``(d, H, W, (D))``
                    probability.append(probability_case)
            else:
                probability = []
                for n_case in range(len(inp)):
                    score_case = score[n_case] # ``(H, W, (D))``
                    score_flatten = score[n_case].flatten() # ``n``
                    inp_case = inp[n_case] # ``(d, H, W, (D))``
                    inp_case = inp_case.reshape(((self.num_class, len(score_flatten)))) # ``(d, n)``
                    inp_case = inp_case.T # ``(n, d)``
                    T = solve_T(inp_case, score_flatten) # return T of shape ``(n, )``
                    probability_case = np.zeros((score_flatten.shape + (self.num_class,))) # ``(n, d)``
                    for ksample in range(len(score)):
                        probability_case[ksample, :] = cal_softmax(inp_case[ksample: ksample + 1], T[ksample]) # ``(1, d)``
                    probability_case = probability_case.T # ``(d, n)``
                    probability_case = probability_case.reshape(((self.num_class,) + score_case.shape)) # ``(d, H, W, (D))``
                    probability.append(probability_case)

        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
        
        return probability
    
    def calculate_probability_appr(self, inp: Union[List[Iterable], np.ndarray], midstage: bool = False, full : bool = False) -> Union[List[Iterable], np.ndarray]:
        """Calculate the calibrated probability with parameters, utilizing approximation of 1-score.
        To acclerate the optimization process, we calculate the probability but divide 1-score equally to other classes.
        I should use it for all the segmentation tasks, as the pixel number is always quite large.

        Note:
            The calibrated probability should have the same dimension with network outputs.
            This is identical to calculate_probability when d == 2, i.e. binary segmentation, which is the most common case. Therefore, I am not very worry about the descrepencies.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
            midstage: If ``True``, return the first calibrated results.
            full: If ``False``, we just put the score in and fill the other domains with 0.

        Returns:
            calibrated_probability: The calibrated probability which would match the accuracy/DSC on validation data, of shape ``(n, d)`` or a list of n ``(d, H, W, (D))``.
        
        """

        if self.extend_param:
            score = self.calibrate(inp, midstage)
        else:
            score = self.calibrate(inp)
            
        if self.mode == "classification":
            # extend from ``(n, )`` to ``(n, d)``
            pred_flatten = np.argmax(inp, axis = 1)
            probability = np.zeros((score.shape + (self.num_class,))) # ``(n, d)``
            if not full or self.estim_algorithm == "atc-model" or (self.estim_algorithm == "ts-atc-model" and midstage == False):
                # ATC cannot get the calibrated probability of all classes, we just fill the other dimension with zeros
                pass
            else:
                # first fill the probability with score / (d-1)
                score_extended = np.tile(score, (self.num_class,1)).transpose()
                probability[np.arange(probability.shape[0]), :] = (1 - score_extended) / (self.num_class - 1) # ``(n, d)``
            
            probability[np.arange(probability.shape[0]), pred_flatten] = score # overwrite the confidence score of the predicted class

        elif self.mode == "segmentation":
            # extend from a list of n ``(H, W, (D))`` to a list of n ``(d, H, W, (D))``
            
            probability = []
            for n_case in range(len(inp)):
                pred_case = np.argmax(inp[n_case], axis = 0) # ``(H, W, (D))``
                #
                pred_flatten = pred_case.flatten() # ``n``
                score_case = score[n_case] # ``(H, W, (D))``
                score_flatten = score[n_case].flatten() # ``n``
                probability_case = np.zeros((score_flatten.shape + (self.num_class,))) # ``(n, d)``
                if not full or self.estim_algorithm == "atc-model" or (self.estim_algorithm == "ts-atc-model" and midstage == False):
                    # ATC cannot get the calibrated probability of all classes, we just fill the other dimension with zeros
                    pass
                else:
                    # first fill the probability with score / (d-1)
                    score_extended = np.tile(score_flatten, (self.num_class,1)).transpose()
                    probability_case[np.arange(probability_case.shape[0]), :] = (1 - score_extended) / (self.num_class - 1) # ``(n, d)``

                probability_case[np.arange(probability_case.shape[0]), pred_flatten] = score_flatten # ``(n, d)``
                probability_case = probability_case.T # ``(d, n)``
                probability_case = probability_case.reshape(((self.num_class,) + score_case.shape)) # ``(d, H, W, (D))``
                probability.append(probability_case)

        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
        
        return probability

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
                score = score - (1 - self.param)
                return np.clip(score, 0, 1)
            elif self.mode == "segmentation":
                score_post = []
                for n_case in range(len(score)):
                    score_case = score[n_case] - (1 - self.param)
                    score_post.append(np.clip(score_case, 0, 1))
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
                return np.clip(score, 0, 1)
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
                    score_post.append(np.clip(score_reshape, 0, 1))
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
            _score = self._normalize(_score)

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