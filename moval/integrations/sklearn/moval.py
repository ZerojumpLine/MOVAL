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
                            confidence_scores = "max_class_probability-conf",
                            estim_algorithm = "ac-model",
                            class_specific = False
                            )

    """


    def __init__(
        self,
        mode: str = "classification",
        confidence_scores: str = "max_class_probability-conf",
        estim_algorithm: str = "ac-model",
        class_specific: bool = False, 
        approximate: bool = False, 
        approximate_boundary: int = 30
    ):
        self.__dict__.update(locals())

    def fit(
        self,
        logits: Union[List[Iterable], np.ndarray],
        gt: Union[List[Iterable], np.ndarray]
    ) -> "MOVAL":
        """Fit the estimator to the given dataset by minimzing the calibration error.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation of shape ``(n, )`` for classification and a list of n ``(H, W, (D))`` for segmentation.
        
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

        if self.estim_algorithm == 'moval-ensemble':
            self.ensemble = True
            moval_models = list(itertools.product(
                ['ts-model', 'doc-model', 'atc-model', 'ts-atc-model'],
                [self.mode],
                moval.models.get_conf_options(),
                [True]))
        elif self.estim_algorithm == 'moval-ensemble-triathlon':
            self.ensemble = True
            # cannot utilize atc model family as they cannot estimate false positives.
            moval_models = list(itertools.product(
                ['ts-model', 'doc-model'],
                [self.mode],
                moval.models.get_conf_options(),
                [True]))
        else:
            self.ensemble = False

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
                solver = moval.solvers.init("base-solver", model = model)
                solver.fit(logits, gt) # model fitting
                if model.extend_param:
                    probability_cond = model.calculate_probability(logits, False)
                else:
                    probability_cond = model.calculate_probability(logits) # ``(n, d)`` or a list of n ``(d, H, W, (D))``
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
            self.fitted_perf = self.get_case_perf(models[0], logits, probability, gt)
            self.ensemble_conds = ensemble_conds

        else:
            model = moval.models.init(
                self.estim_algorithm,
                mode = self.mode,
                num_class = self.numclass,
                confidence_scores = self.confidence_scores,
                class_specific = self.class_specific
                )
            solver = moval.solvers.init("base-solver", model = model)
            solver.fit(logits, gt) # model fitting
            probability = model.calculate_probability(logits)

            # save the results to self attributes.
            self.model_ = model
            self.solver_ = solver
            self.fitted_perf = self.get_case_perf(model, logits, probability, gt)

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
                      inp: Union[List[Iterable], np.ndarray],
                      probability: Union[List[Iterable], np.ndarray],
                      gt: Union[List[Iterable], np.ndarray]):
        """Store the estimated results of n fitted data.

        Note:
            For segmentaiton tasks, we only save the average dsc of all the foreground classes.

        Args:
            model: The fitted moval model.
            inp: The network output (logits) of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            probability: The predicted probability of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation of shape ``(n, )`` for classification and a list of n ``(H, W, (D))`` for segmentation.

        Returns:
            fitted_perf: a list contain estimated results of shape of len n.
        
        """

        fitted_perf = []
        if model.mode == "classification":
            # probability of shape ``(n, d)```
            for n_case in range(len(probability)):
                estim_acc = np.max(probability[n_case])
                fitted_perf.append(estim_acc)
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
            # logits is a list of n ``(d, H, W, (D))``
            for n_case in range(len(probability)):
                estim_dsc = model.estimate_F1score(inp, probability, gt_guide = gt_guide)
                fitted_perf.append(np.mean(estim_dsc[1:]))
        
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
        """Estimate model performance using logits.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
            gt_guide: A numpy array of size ``(n, d)`` for segmentation, indicating the existing of object d in sample n.
                If ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim: estimated accuracy (float) for classification tasks, 
                or estimated foreground dsc of shape ``(d-1, )`` for segmentation tasks.

        Example:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> estim_acc = moval_model.estimate(logits)

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
            
            probabilities = []
            for model in self.model_:
                if model.extend_param:
                    probability_cond = model.calculate_probability(logits, False)
                else:
                    probability_cond = model.calculate_probability(logits)
                probabilities.append(probability_cond)
                probability = self.probability_aggregation(probabilities)
            
            if model.mode == "classification":
                estim_acc = np.mean(np.max(probability, axis = 1))
                return estim_acc
            else:
                estim_dsc = model.estimate_F1score(logits, probability, gt_guide = gt_guide)
                return estim_dsc[1:]
        else:
            model = self.model_

            if model.mode == "classification":
                estim_acc, estim_acc_allcls = model(logits)
                return estim_acc
            else:
                estim_acc, estim_dsc = model(logits, gt_guide = gt_guide)
                return estim_dsc[1:]
    
    def estimate_sensitivity(self,
                 logits: Union[List[Iterable], np.ndarray],
                 gt_guide: np.ndarray = None):
        """Estimate model's sensitivity using logits.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
            gt_guide: A numpy array of size ``(n, d)`` for segmentation, indicating the existing of object d in sample n.
                If ``False``, it means that there isn't any manuel label of class d in this sample.
        
        Returns:
            estim: estimated average sensitivity (float) for classification tasks, 
                or estimated sensitivity of shape ``(d-1, )`` for segmentation tasks.

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
            probabilities = []
            for model in self.model_:
                if model.extend_param:
                    probability_cond = model.calculate_probability(logits, False)
                else:
                    probability_cond = model.calculate_probability(logits)

                probabilities.append(probability_cond)
                probability = self.probability_aggregation(probabilities)
        else:
            model = self.model_
            if model.extend_param:
                probability = model.calculate_probability(logits, False)
            else:
                probability = model.calculate_probability(logits)

        if model.mode == "classification":
            estim_sensitivity = model.estimate_sensitivity(logits, probability)
            return np.mean(estim_sensitivity)
        else:
            estim_sensitivity = model.estimate_sensitivity(logits, probability, gt_guide = gt_guide)
            return estim_sensitivity[1:]
    
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
            probabilities = []
            for model in self.model_:
                if model.extend_param:
                    probability_cond = model.calculate_probability(logits, False)
                else:
                    probability_cond = model.calculate_probability(logits)

                probabilities.append(probability_cond)
                probability = self.probability_aggregation(probabilities)
        else:
            model = self.model_
            if model.extend_param:
                probability = model.calculate_probability(logits, False)
            else:
                probability = model.calculate_probability(logits)

        if model.mode == "classification":
            estim_precision = model.estimate_precision(logits, probability)
            return np.mean(estim_precision)
        else:
            estim_precision = model.estimate_precision(logits, probability, gt_guide = gt_guide)
            return estim_precision[1:]
    
    def estimate_F1score(self,
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
            probabilities = []
            for model in self.model_:
                if model.extend_param:
                    probability_cond = model.calculate_probability(logits, False)
                else:
                    probability_cond = model.calculate_probability(logits)

                probabilities.append(probability_cond)
                probability = self.probability_aggregation(probabilities)
        else:
            model = self.model_
            if model.extend_param:
                probability = model.calculate_probability(logits, False)
            else:
                probability = model.calculate_probability(logits)

        if model.mode == "classification":
            estim_F1score = model.estimate_F1score(logits, probability)
            return np.mean(estim_F1score)
        else:
            estim_F1score = model.estimate_F1score(logits, probability, gt_guide = gt_guide)
            return estim_F1score[1:]

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
            probabilities = []
            for model in self.model_:
                if model.extend_param:
                    probability_cond = model.calculate_probability(logits, False)
                else:
                    probability_cond = model.calculate_probability(logits)

                probabilities.append(probability_cond)
                probability = self.probability_aggregation(probabilities)
        else:
            model = self.model_
            if model.extend_param:
                probability = model.calculate_probability(logits, False)
            else:
                probability = model.calculate_probability(logits)

        if model.mode == "classification":
            estim_AUC = model.estimate_AUC(probability)
            return np.mean(estim_AUC)
        else:
            estim_AUC = model.estimate_AUC(probability, gt_guide = gt_guide)
            return estim_AUC[1:]
    
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
            name_to_save = filename[:-3]
            ckpts = []
            # A ckpt saving the meta-info
            ckpt = {}
            ckpt["mode"] = model.mode
            ckpt["estim_algorithm"] = self.estim_algorithm
            ckpt["mode"] = self.mode
            ckpt["confidence_scores"] = self.confidence_scores
            ckpt["class_specific"] = self.class_specific
            ckpt["ensemble"] == self.ensemble
            ckpt["ensemble_conds"] = self.ensemble_conds
            # Save the object as a pickle file
            with open(filename_to_save, 'wb') as file:
                pickle.dump(ckpt, file)
            ckpts.append(ckpt)
            for model, ensemble_cond in zip(self.model_, self.ensemble_conds):

                ckpt = {}

                ckpt["estim_algorithm"] = model.estim_algorithm
                ckpt["mode"] = model.mode
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
            
            ckpt["ensemble"] == self.ensemble
            ckpt["estim_algorithm"] = self.model_.estim_algorithm
            ckpt["mode"] = self.model_.mode
            ckpt["num_class"] = self.model_.num_class
            ckpt["confidence_scores"] = self.model_.confidence_scores
            ckpt["class_specific"] = self.model_.class_specific
            ckpt["param"] = self.model_.param
            ckpt["n_dim_"] = self.n_dim_
            ckpt["fitted_perf"] = self.fitted_perf
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

            name_to_load = filename[:-3]
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

        return moval_model
    
# Dynamically remove the set_fit_request method
del MOVAL.set_fit_request