"""Define the MOVAL model."""
import numpy as np
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union
from sklearn.base import BaseEstimator
import itertools
import moval.models
import moval.solvers
import pickle

class MOVAL(BaseEstimator):
    """MOVAL model defined as part of a ``scikit-learn``-like API.

    Attributes:
        mode (str):
            The given task to estimate model performance. |Default:| ``classification``
        metric (str):
            The performance metric we optimize moval model to align. |Default:| ``accuracy``
            Metric can be ``accuracy`` | ``sensitivity`` | ``precision`` | ``f1score`` | ``auc``
            This is only effective when class_specific is ``True``, otherwise moval will always align overall accuracy.
        confidence_scores (str):
            The method to calculate the confidence scores. We provide a list of confidence score calculation methods which 
            can be displayed by running :py:func:`moval.models.get_conf_options`. |Default:| ``max_class_probability-conf``
        estim_algorithm (str):
            The algorithm to estimate model performance. We also provide a list of estimation algorithm which can be displayed by
            running :py:func:`moval.models.get_estim_options`. |Default:| ``ac-model``
        class_specific (bool):
            If ``True``, the calculation will match class-wise confidence to class-wise accuracy. |Default:| ``False``
        approximate (bool):
            If ``True``, we crop the image and label map to accelerate the optimization of segmentation. |Default:| ``False``
        approximate_boundary (int):
            The enlarged regions of the region to crop. |Default:| ``30``

    Example:
        
        >>> import moval
        >>> moval_model = moval.MOVAL(
                            mode = "classification",
                            metric = "accuracy",
                            confidence_scores = "max_class_probability-conf",
                            estim_algorithm = "ac-model",
                            class_specific = False
                            )

    """


    def __init__(
        self,
        mode: str = "classification",
        metric: str = "accuracy",
        confidence_scores: str = "max_class_probability-conf",
        estim_algorithm: str = "ac-model",
        class_specific: bool = False, 
        approximate: bool = False, 
        approximate_boundary: int = 30
    ):
        self.__dict__.update(locals())

        # decide if the model is in the ensemble mode
        if self.estim_algorithm.split('-')[1] == 'ensemble':
            self.ensemble = True
        else:
            self.ensemble = False

    def fit(
        self,
        logits: Union[List[Iterable], np.ndarray],
        gt: Union[List[Iterable], np.ndarray], 
        batch: int = 1
    ) -> "MOVAL":
        """Fit the estimator to the given dataset by minimzing the calibration error.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation of shape ``(n, )`` for classification and a list of n ``(H, W, (D))`` for segmentation.
            batch: To match the group-wise accuracy with group-wise confidence score. batch is the group size. This is useful when validation data per case is few.
        
        Note:
            It only makes sense if the class frequency is similar between class index to use ``batch`` > 1!
        
        Return:
            ``self``

        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)

        """

        # Input validation
        if self.mode == "classification":
            if logits.shape[0] != gt.shape[0]:
                raise ValueError(
                        f"Invalid input samples: logits and GT should represent the same number of samples"
                        f"(n_samples of logits, {logits.shape[0]}, while n_samples of GT , {gt.shape[0]})."
                    )

            if len(logits.shape) != (len(gt.shape) + 1):
                raise ValueError(
                        f"Invalid input dimension: logits should have one additional dimension than GT"
                        f"(dimension of logits, {len(logits.shape)}, while dimension of GT , {len(gt.shape)})."
                    )
            
            if len(gt.shape) != 1:
                raise ValueError(
                    f"Invalid input dimension for classification: GT should have the dimension of 1"
                    f"(dimension of GT, {len(logits.shape)})."
                )

        else:
            if len(logits) != len(gt):
                raise ValueError(
                        f"Invalid input samples: logits and GT should represent the same number of samples"
                        f"(n_samples of logits, {len(logits)}, while n_samples of GT , {len(gt)})."
                    )

            if len(logits[0].shape) != (len(gt[0].shape) + 1):
                raise ValueError(
                        f"Invalid input dimension: logits should have one additional dimension than GT"
                        f"(dimension of logits, {len(logits[0].shape)}, while dimension of GT , {len(gt[0].shape)})."
                    )
            
            if (len(gt[0].shape) != 2) and (len(gt[0].shape) != 3):
                raise ValueError(
                    f"Invalid input dimension for segmentation: GT should have the dimension of 2 or 3"
                    f"(dimension of GT, {len(gt[0].shape)})."
                )

        if self.mode == "segmentation" and self.approximate == True:
            logits, gt = self.crop(logits = logits, gt = gt, approximate_boundary = self.approximate_boundary)
        
        # initilization
        if isinstance(logits, list):
            self.numclass = logits[0].shape[0]
        else:
            self.numclass = logits.shape[-1]

        if self.estim_algorithm == "moval-ensemble":
            # All the CS variants
            moval_models = list(itertools.product(
                ["ts-model", "doc-model", "atc-model", "ts-atc-model"],
                [self.mode],
                moval.models.get_conf_options(),
                [True]))
        elif self.estim_algorithm == "moval-ensemble-triathlon":
            # All the CS variants that support the estimation of all metrics
            # cannot utilize atc model family as they cannot estimate false positives.
            moval_models = list(itertools.product(
                ["ts-model", "doc-model"],
                [self.mode],
                moval.models.get_conf_options(),
                [True]))
        elif self.estim_algorithm == "moval-ensemble-debug":
            # just for debuging, aggregating 2 conditions.
            moval_models = []
            moval_models.append(["ts-model", self.mode, "energy-conf", True])
            moval_models.append(["doc-model", self.mode, "max_class_probability-conf", False])


        if self.ensemble:
            ensemble_conds = []
            models = []
            solvers = []
            probabilities = []
            for estim_algorithm, mode, confidence_scores, class_specific in moval_models:
                model = moval.models.init(
                    estim_algorithm,
                    mode = mode,
                    num_class = self.numclass,
                    confidence_scores = confidence_scores,
                    class_specific = class_specific
                )

                # Zeju Li, 16 May 2024: The leads to weird behaviour of precision estimation, skip it now.
                # if class_specific == True:
                #     # find a good initatlization, as other metrics would also depend on non-maximum logits
                #     # also find a good init for accuary of sensitivty, to handle the corner cases, where making no prediction for specific classes.
                #     model_pre = moval.models.init(
                #         estim_algorithm,
                #         mode = self.mode,
                #         num_class = self.numclass,
                #         confidence_scores = self.confidence_scores,
                #         class_specific = False
                #         )
                #     solver_pre = moval.solvers.init("base-solver", model = model_pre, metric = "accuracy")
                #     solver_pre.fit(logits, gt)
                #     model.param = np.ones(self.numclass) * model_pre.param

                solver = moval.solvers.init("base-solver", model = model, metric = self.metric)
                solver.fit(logits, gt, batch) # model fitting
                if self.metric == "precision" or self.metric == "auc":
                    probability_cond = model.calculate_probability(logits, appr = True, full = True) # ``(n, d)`` or a list of n ``(d, H, W, (D))``
                else:
                    probability_cond = model.calculate_probability(logits, appr = True) # ``(n, d)`` or a list of n ``(d, H, W, (D))``
                #
                models.append(model)
                solvers.append(solver)
                probabilities.append(probability_cond)
                ensemble_cond = f"estim_algorithm_{estim_algorithm}_mode_{mode}_confidence_scores_{confidence_scores}_class_specific_{class_specific}"
                ensemble_conds.append(ensemble_cond)
            #
            # aggregate probabilities
            probability = self.probability_aggregation(probabilities)
            self.model_ = models
            self.solver_ = solvers
            print(f"Calculating and saving the fitted case-wise performance...")
            self.fitted_perf = self.get_case_perf(models[0], self.metric, logits, probability, gt)
            self.ensemble_conds = ensemble_conds

        else:
            model = moval.models.init(
                self.estim_algorithm,
                mode = self.mode,
                num_class = self.numclass,
                confidence_scores = self.confidence_scores,
                class_specific = self.class_specific
                )
            
            # Zeju Li, 16 May 2024: The leads to weird behaviour of precision estimation, skip it now.
            # if self.class_specific == True:
            #     # find a good initatlization, as other metrics would also depend on non-maximum logits
            #     # also find a good init, to handle the corner cases, where making no prediction for specific classes.
            #     model_pre = moval.models.init(
            #         self.estim_algorithm,
            #         mode = self.mode,
            #         num_class = self.numclass,
            #         confidence_scores = self.confidence_scores,
            #         class_specific = False
            #         )
            #     solver_pre = moval.solvers.init("base-solver", model = model_pre, metric = "accuracy")
            #     solver_pre.fit(logits, gt)
            #     model.param = np.ones(self.numclass) * model_pre.param

            solver = moval.solvers.init("base-solver", model = model, metric = self.metric)
            solver.fit(logits, gt, batch) # model fitting

            # save the results to self attributes.
            self.model_ = model
            self.solver_ = solver
            print(f"Calculating and saving the fitted case-wise performance...")
            self.fitted_perf = self.get_case_perf(model, self.metric, logits, gt)

        if self.mode == "classification":
            self.n_dim_ = len(logits.shape)
        else:
            self.n_dim_ = len(logits[0].shape)

        return self

    @classmethod
    def probability_aggregation(cls,
                                probabilities):
        """Calcualte the mean of multiple probabilities.

        Args:
            probabilities: The predicted probability.
                This is a list of k ``(n, d)`` for clasfication and a list of k lists of n ``(d, H, W, (D))`` for segmentation.
        
        Returns:
            probability_agg: The average probability of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 

        """
        if len(probabilities[0].shape) == 2:
            # classification
            probability_agg = np.mean(np.array(probabilities), axis = 0)
        else:
            # segmentation
            probability_agg = []
            for n_case in len(probabilities[0]):
                for k_cond in len(probabilities):
                    probabilities_case = []
                    probabilities_case.append(probabilities[k_cond][n_case])
                probability_agg.append(np.mean(np.array(probabilities_case)), axis = 0)
        
        return probability_agg

    @classmethod
    def get_case_perf(cls,
                      model: moval.models,
                      metric: str,
                      inp: Union[List[Iterable], np.ndarray],
                      gt: Union[List[Iterable], np.ndarray]):
        """Store the estimated results of n fitted data.

        Note:
            For segmentaiton tasks, we only save the average dsc of all the foreground classes.

        Args:
            model: The fitted moval model.
            metric: The performance metric to follow.
            inp: The network output (logits) of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation of shape ``(n, )`` for classification and a list of n ``(H, W, (D))`` for segmentation.

        Returns:
            fitted_perf: a list contain estimated results of shape of len n. Each element is a scalar if estimated accuracy, otherwise ``(d, )``.
        
        """

        fitted_perf = []
        
        if metric == "precision" or metric == "auc":
            probability = model.calculate_probability(inp, appr = True, full=True)
        else:
            probability = model.calculate_probability(inp, appr = True)

        if model.mode == "classification":
            
            for n_case in range(len(inp)):

                if metric == "accuracy":
                    estim_perf_case, _ = model.estimate_accuracy(inp[n_case:n_case + 1])
                elif metric == "sensitivity":
                    estim_perf_case = model.estimate_sensitivity(inp[n_case:n_case + 1], probability[n_case:n_case + 1])
                elif metric == "precision":
                    estim_perf_case = model.estimate_precision(probability[n_case:n_case + 1])
                elif metric == "f1score":
                    estim_perf_case = model.estimate_f1score(inp[n_case:n_case + 1], probability[n_case:n_case + 1])
                elif metric == "auc":
                    estim_perf_case = model.estimate_auc(probability[n_case:n_case + 1])
                else:
                    ValueError(f"Unsupported metric '{metric}'") 
                
                fitted_perf.append(estim_perf_case)
        
        else:

            # generate gt_guide from gt here.
            gt_guide = []
            for n_case in range(len(gt)):
                gt_case     = gt[n_case]
                gt_exist = []
                for k_cls in range(model.num_class):
                    gt_exist.append(np.sum(gt_case == k_cls) > 0)
                gt_guide.append(gt_exist)
            gt_guide = np.array(gt_guide)

            for n_case in range(len(inp)):
                
                if metric == "accuracy":
                    estim_perf_case, _ = model.estimate_accuracy(inp[n_case:n_case + 1], gt_guide = gt_guide[n_case:n_case + 1])
                elif metric == "sensitivity":
                    estim_perf_case = model.estimate_sensitivity(inp[n_case:n_case + 1], probability[n_case:n_case + 1], gt_guide = gt_guide[n_case:n_case + 1])
                elif metric == "precision":
                    estim_perf_case = model.estimate_precision(probability[n_case:n_case + 1], gt_guide = gt_guide[n_case:n_case + 1])
                elif metric == "f1score":
                    estim_perf_case = model.estimate_f1score(inp[n_case:n_case + 1], probability[n_case:n_case + 1], gt_guide = gt_guide[n_case:n_case + 1])
                elif metric == "auc":
                    estim_perf_case = model.estimate_auc(probability[n_case:n_case + 1], gt_guide = gt_guide[n_case:n_case + 1])
                else:
                    ValueError(f"Unsupported metric '{metric}'")
                
                fitted_perf.append(estim_perf_case)
        
        return fitted_perf

    @classmethod
    def crop(cls,
             logits: List[Iterable],
             gt: List[Iterable],
             approximate_boundary):
        """Crop the image and label map to accelrate the optimization process.
        
        Here we do the cropping based on the label map (gt). We first find the bounding box and enlarge the region with boundary = 30 pixels.

        Args:
            logits: The network output (logits) of a list of n ``(d, H, W, (D))``. 
            gt: The cooresponding annotation of a list of n ``(H, W, (D))``.
            approximate_boundary: The enlarged regions of the region to crop.
        
        Return:
            logits_post: The cropped network output (logits) of a list of n ``(d, H', W', (D'))``. 
            gt_post: The cooresponding cropped annotation of a list of n ``(H', W', (D'))``.

        """

        logits_post = []
        gt_post = []
        for k_case in range(len(logits)):
            logit_map = logits[k_case]
            gt_map = gt[k_case]
        
            if len(gt_map.shape) == 2:
                border_x, border_y = gt_map.shape
                ind_x, ind_y = np.where(gt_map != 0)
                #
                max_x = np.min((np.max(ind_x) + approximate_boundary, border_x))
                min_x = np.max((np.min(ind_x) - approximate_boundary, 0))
                #
                max_y = np.min((np.max(ind_y) + approximate_boundary, border_y))
                min_y = np.max((np.min(ind_y) - approximate_boundary, 0))
                #
                logits_post.append(logit_map[:, min_x:max_x, min_y:max_y])
                gt_post.append(gt_map[min_x:max_x, min_y:max_y])

            elif len(gt_map.shape) == 3:
                border_x, border_y, border_z = gt_map.shape
                ind_x, ind_y, ind_z = np.where(gt_map != 0)
                #
                max_x = np.min((np.max(ind_x) + approximate_boundary, border_x))
                min_x = np.max((np.min(ind_x) - approximate_boundary, 0))
                #
                max_y = np.min((np.max(ind_y) + approximate_boundary, border_y))
                min_y = np.max((np.min(ind_y) - approximate_boundary, 0))
                #
                max_z = np.min((np.max(ind_z) + approximate_boundary, border_z))
                min_z = np.max((np.min(ind_z) - approximate_boundary, 0))
                #
                logits_post.append(logit_map[:, min_x:max_x, min_y:max_y, min_z:max_z])
                gt_post.append(gt_map[min_x:max_x, min_y:max_y, min_z:max_z])
            else:
                raise ValueError("Not implemented!")
        
        return logits_post, gt_post

    def estimate(self,
                 logits: Union[List[Iterable], np.ndarray],
                 gt_guide: np.ndarray = None):
        """The function to estimate the model performance. This will call other corresponding metric calculation functions.
        
        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
            gt_guide: A numpy array of size ``(n, d)`` for segmentation, indicating the existing of object d in sample n.
                If ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim: estimated metric (float) for classification tasks, 
                or class-wise estimated metric of shape ``(d-1, )`` for segmentation tasks.

        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> estim_acc = moval_model.estimate(logits)

        """

        if self.metric == "accuracy":
            return self.estimate_accuracy(logits, gt_guide)
        elif self.metric == "sensitivity":
            return self.estimate_sensitivity(logits, gt_guide)
        elif self.metric == "precision":
            return self.estimate_precision(logits, gt_guide)
        elif self.metric == "f1score":
            return self.estimate_f1score(logits, gt_guide)
        elif self.metric == "auc":
            return self.estimate_auc(logits, gt_guide)
        else:
            ValueError(f"Unsupported metric '{self.metric}'")

    def get_probability(self,
                        logits: Union[List[Iterable], np.ndarray]):
        """Obtain the the probability give the logits.

        Note:
            It will ensemble (not exactly ensemble, but calculate the mean of probability) moval-ensemble.
            Here we utilize ``calculate_probability`` for classification, while ``calculate_probability_appr`` for segmentation. 
            Because the sample number of segmentation tasks is too large.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
        
        Examples:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> moval_model = moval.MOVAL(confidence_scores = "moval-ensemble")
            >>> probability = moval_model.get_probability(logits)
        
        """
            
        if self.ensemble:
            probabilities = []
            for model in self.model_:
                if self.metric == "precision" or self.metric == "auc":
                    probability_cond = model.calculate_probability(logits, appr = True, full=True)
                else:
                    probability_cond = model.calculate_probability(logits, appr = True)

                probabilities.append(probability_cond)
            probability = self.probability_aggregation(probabilities)
        else:
            model = self.model_
            if self.metric == "precision" or self.metric == "auc":
                probability = model.calculate_probability(logits, appr = True, full=True)
            else:
                probability = model.calculate_probability(logits, appr = True)

        return probability

    def estimate_accuracy(self,
                 logits: Union[List[Iterable], np.ndarray],
                 gt_guide: np.ndarray = None):
        """Estimate accuracy using logits.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
            gt_guide: A numpy array of size ``(n, d)`` for segmentation, indicating the existing of object d in sample n.
                If ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim: estimated accuracy (float). 

        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> estim_acc = moval_model.estimate_accuracy(logits)

        """

        # Input validation
        if self.mode == "classification":
            if self.n_dim_ != len(logits.shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits.shape)})"
                    )
        else:
            if self.n_dim_ != len(logits[0].shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits[0].shape)})"
                    )

        if self.ensemble:
            
            probability = self.get_probability(logits)

            if self.mode == "classification":
                estim_acc = np.mean(np.max(probability, axis = 1))
                return estim_acc
            else:
                scores = []
                for n_case in range(len(probability)):
                    score_flatten = np.max(probability[n_case], axis = 0).flatten() # ``n``
                    scores.append(score_flatten)
                estim_acc = np.mean( np.concatenate(scores) )
                
                return estim_acc
        else:
            model = self.model_

            estim_acc, _ = model.estimate_accuracy(logits)
            return estim_acc
    
    def estimate_sensitivity(self,
                 logits: Union[List[Iterable], np.ndarray],
                 gt_guide: np.ndarray = None):
        """Estimate model's sensitivity using logits.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
            gt_guide: A numpy array of size ``(n, d)`` for segmentation, indicating the existing of object d in sample n.
                If ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim: estimated sensitivity of shape ``(d, )``.
        
        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> estim_sensitivity = moval_model.estimate_sensitivity(logits)

        """

        # Input validation
        if self.mode == "classification":
            if self.n_dim_ != len(logits.shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits.shape)})"
                    )
        else:
            if self.n_dim_ != len(logits[0].shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits[0].shape)})"
                    )

        probability = self.get_probability(logits)
        if self.ensemble:
            estim_sensitivity = self.model_[0].estimate_sensitivity(logits, probability, gt_guide = gt_guide)
        else:
            estim_sensitivity = self.model_.estimate_sensitivity(logits, probability, gt_guide = gt_guide)

        return estim_sensitivity
    
    def estimate_precision(self,
                 logits: Union[List[Iterable], np.ndarray],
                 gt_guide: np.ndarray = None):
        """Estimate model's sensitivity using logits.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
            gt_guide: A numpy array of size ``(n, d)`` for segmentation, indicating the existing of object d in sample n.
                If ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim: estimated average precision (float) for classification tasks, 
                or estimated precision of shape ``(d-1, )`` for segmentation tasks.

        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> estim_precision = moval_model.estimate_precision(logits)

        """

        # Input validation
        if self.mode == "classification":
            if self.n_dim_ != len(logits.shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits.shape)})"
                    )
        else:
            if self.n_dim_ != len(logits[0].shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits[0].shape)})"
                    )

        probability = self.get_probability(logits)
        if self.ensemble:
            estim_precision = self.model_[0].estimate_precision(probability, gt_guide = gt_guide)
        else:
            estim_precision = self.model_.estimate_precision(probability, gt_guide = gt_guide)

        return estim_precision
    
    def estimate_f1score(self,
                 logits: Union[List[Iterable], np.ndarray],
                 gt_guide: np.ndarray = None):
        """Estimate model's F1score using logits.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
            gt_guide: A numpy array of size ``(n, d)`` for segmentation, indicating the existing of object d in sample n.
                If ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim: estimated average F1score (float) for classification tasks, 
                or estimated dsc of shape ``(d-1, )`` for segmentation tasks.

        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> estim_f1score = moval_model.estimate_f1score(logits)
                
        """

        # Input validation
        if self.mode == "classification":
            if self.n_dim_ != len(logits.shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits.shape)})"
                    )
        else:
            if self.n_dim_ != len(logits[0].shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits[0].shape)})"
                    )

        probability = self.get_probability(logits)
        if self.ensemble:
            estim_f1score = self.model_[0].estimate_f1score(logits, probability, gt_guide = gt_guide)
        else:
            estim_f1score = self.model_.estimate_f1score(logits, probability, gt_guide = gt_guide)

        return estim_f1score

    def estimate_auc(self,
                 logits: Union[List[Iterable], np.ndarray],
                 gt_guide: np.ndarray = None):
        """Estimate model's AUC using logits.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
            gt_guide: A numpy array of size ``(n, d)`` for segmentation, indicating the existing of object d in sample n.
                If ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim: estimated average AUC (float) for classification tasks, 
                or estimated AUC of shape ``(d-1, )`` for segmentation tasks.

        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> estim_auc = moval_model.estimate_auc(logits)
                
        """

        # Input validation
        if self.mode == "classification":
            if self.n_dim_ != len(logits.shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits.shape)})"
                    )
        else:
            if self.n_dim_ != len(logits[0].shape):
                raise ValueError(
                        f"Inconsistent input dimension: training and test samples have different dimenstions"
                        f"(dimension of training samples, {len(self.n_dim_)}, while dimension of test samples, {len(logits[0].shape)})"
                    )

        probability = self.get_probability(logits)
        if self.ensemble:
            estim_auc = self.model_[0].estimate_auc(probability, gt_guide = gt_guide)
        else:
            estim_auc = self.model_.estimate_auc(probability, gt_guide = gt_guide)

        return estim_auc
    
    def save(self, filename: str):
        """Save current parameters to disk using pickle.

        Args:
            filename: The path to the file in which to save the parameters.
        
        Return:
            The saved parameters, or a list of saved paramters when requiring model ensembling.

        Examples:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> moval_model.save('./foo.pkl')

        """

        if self.ensemble:
            # Object or variable to be saved
            name_to_save = filename[:-4]
            ckpts = []
            # A ckpt saving the meta-info
            ckpt = {}
            ckpt["mode"] = self.mode
            ckpt["metric"] = self.metric
            ckpt["estim_algorithm"] = self.estim_algorithm
            ckpt["mode"] = self.mode
            ckpt["confidence_scores"] = self.confidence_scores
            ckpt["class_specific"] = self.class_specific
            ckpt["ensemble"] = self.ensemble
            ckpt["ensemble_conds"] = self.ensemble_conds
            # Save the object as a pickle file
            with open(filename, 'wb') as file:
                pickle.dump(ckpt, file)
            ckpts.append(ckpt)
            for model, ensemble_cond in zip(self.model_, self.ensemble_conds):

                ckpt = {}

                ckpt["estim_algorithm"] = model.estim_algorithm
                ckpt["mode"] = model.mode
                ckpt["metric"] = self.metric
                ckpt["num_class"] = model.num_class
                ckpt["confidence_scores"] = model.confidence_scores
                ckpt["class_specific"] = model.class_specific
                ckpt["param"] = model.param
                ckpt["n_dim_"] = self.n_dim_
                ckpt["fitted_perf"] = self.fitted_perf
                if model.extend_param:
                    ckpt["param_ext"] = model.param_ext
                if model.conf.normalization:
                    ckpt["max_value"] = model.max_value
                    ckpt["min_value"] = model.min_value

                filename_to_save = f"{name_to_save}_{ensemble_cond}.pkl"

                # Save the object as a pickle file
                with open(filename_to_save, 'wb') as file:
                    pickle.dump(ckpt, file)

                ckpts.append(ckpt)

            return ckpts
        
        else:
            # Object or variable to be saved
            ckpt = {}
            
            ckpt["estim_algorithm"] = self.model_.estim_algorithm
            ckpt["mode"] = self.model_.mode
            ckpt["metric"] = self.metric
            ckpt["num_class"] = self.model_.num_class
            ckpt["confidence_scores"] = self.model_.confidence_scores
            ckpt["class_specific"] = self.model_.class_specific
            ckpt["param"] = self.model_.param
            ckpt["n_dim_"] = self.n_dim_
            ckpt["fitted_perf"] = self.fitted_perf
            ckpt["ensemble"] = self.ensemble
            if self.model_.extend_param:
                ckpt["param_ext"] = self.model_.param_ext
            if self.model_.conf.normalization:
                ckpt["max_value"] = self.model_.max_value
                ckpt["min_value"] = self.model_.min_value

            # Save the object as a pickle file
            with open(filename, 'wb') as file:
                pickle.dump(ckpt, file)

            return ckpt
    
    @classmethod
    def load(cls, filename: str) -> "MOVAL":
        """load parameters from disk.

        Args:
            filename: The path to the file in which to save the parameters.
        
        Return:
            The loaded model, or a list of loaded models when requiring model ensembling.
        
        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> moval_model.save('./foo.pkl')
            >>> loaded_model = moval.MOVAL.load('./foo.pkl')
            >>> estim_acc = loaded_model.estimate(logits)

        """

        with open(filename, 'rb') as file:
            loaded_ckpt = pickle.load(file)
        
        moval_model = moval.MOVAL(
                            mode = loaded_ckpt["mode"],
                            confidence_scores = loaded_ckpt["confidence_scores"],
                            estim_algorithm = loaded_ckpt["estim_algorithm"],
                            class_specific = loaded_ckpt["class_specific"]
                            )
        
        moval_model.ensemble = loaded_ckpt["ensemble"]

        if moval_model.ensemble:

            name_to_load = filename[:-4]
            moval_model.ensemble_conds = loaded_ckpt["ensemble_conds"]
            models = []
            for ensemble_cond in moval_model.ensemble_conds:
                filename_to_load = f"{name_to_load}_{ensemble_cond}.pkl"
                with open(filename_to_load, 'rb') as file:
                    loaded_ckpt = pickle.load(file)

                model = moval.models.init(
                    loaded_ckpt["estim_algorithm"],
                    mode = loaded_ckpt["mode"],
                    num_class = loaded_ckpt["num_class"],
                    confidence_scores = loaded_ckpt["confidence_scores"],
                    class_specific = loaded_ckpt["class_specific"]
                    )

                model.param = loaded_ckpt["param"]
                if model.extend_param:
                    model.param_ext = loaded_ckpt["param_ext"]
                
                if model.conf.normalization:
                    model.max_value = loaded_ckpt["max_value"]
                    model.min_value = loaded_ckpt["min_value"]

                model.is_fitted = True
                models.append(model)

                moval_model.model_ = models

        else:

            model = moval.models.init(
                loaded_ckpt["estim_algorithm"],
                mode = loaded_ckpt["mode"],
                num_class = loaded_ckpt["num_class"],
                confidence_scores = loaded_ckpt["confidence_scores"],
                class_specific = loaded_ckpt["class_specific"]
                )

            model.param = loaded_ckpt["param"]
            if model.extend_param:
                model.param_ext = loaded_ckpt["param_ext"]
            
            if model.conf.normalization:
                model.max_value = loaded_ckpt["max_value"]
                model.min_value = loaded_ckpt["min_value"]

            model.is_fitted = True

            moval_model.model_ = model

        moval_model.n_dim_ = loaded_ckpt["n_dim_"]
        moval_model.fitted_perf = loaded_ckpt["fitted_perf"]
        moval_model.metric = loaded_ckpt["metric"]

        return moval_model
    
# Dynamically remove the set_fit_request method
del MOVAL.set_fit_request