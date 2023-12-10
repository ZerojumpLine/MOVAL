"""Define the MOVAL model."""
import numpy as np
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union
from sklearn.base import BaseEstimator
import moval.models
import moval.solvers
import pickle

class MOVAL(BaseEstimator):
    """ MOVAL model defined as part of a ``scikit-learn``-like API.

    Attributes:
        mode (str):
            The given task to estimate model performance. |Default:| ``classification``
        confidence_scores (str):
            The method to calculate the confidence scores. We provide a list of confidence score calculation methods which 
            can be displayed by running :py:func:`moval.models.get_conf_options`. |Default:| ``raw``
        estim_algorithm (str):
            The algorithm to estimate model performance. We also provide a list of estimation algorithm which can be displayed by
            running :py:func:`moval.models.get_estim_options`. |Default:| ``ac``
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

        # initilization
        if isinstance(logits, list):
            self.numclass = logits[0].shape[0]
        else:
            self.numclass = logits.shape[-1]

        model = moval.models.init(
            self.estim_algorithm,
            mode = self.mode,
            num_class = self.numclass,
            confidence_scores = self.confidence_scores,
            class_specific = self.class_specific
            )
        solver = moval.solvers.init("base-solver", model = model)

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

        solver.fit(logits, gt)

        #
        self.model_ = model
        self.solver_ = solver
        if self.mode == "classification":
            self.n_dim_ = len(logits.shape)
        else:
            self.n_dim_ = len(logits[0].shape)

        return self

    @classmethod
    def crop(self,
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
                 logits: Union[List[Iterable], np.ndarray]):
        """Estimate model performance using logits.

        Args:
            logits: A numpy array of size ``(n, d)`` for classification or a list of n ``(d, H, W, (D))`` for segmentation.
        
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

        model = self.model_

        if model.mode == "classification":
            estim_acc, estim_acc_allcls = model(logits)
            return estim_acc
        else:
            estim_acc, estim_dsc = model(logits)
            return estim_dsc[1:]
    
    def save(self, filename: str):
        """Save current parameters to disk using pickle.

        Args:
            filename: The path to the file in which to save the parameters.
        
        Return:
            The saved parameters.

        Examples:

            >>> import moval
            >>> import numpy as np
            >>> logits = np.random.randn(1000, 10)
            >>> gt = np.random.randint(0, 10, (1000))
            >>> moval_model = moval.MOVAL()
            >>> moval_model.fit(logits, gt)
            >>> moval_model.save('./foo.pkl')

        """
        # Object or variable to be saved
        ckpt = {}

        ckpt["estim_algorithm"] = self.model_.estim_algorithm
        ckpt["mode"] = self.model_.mode
        ckpt["num_class"] = self.model_.num_class
        ckpt["confidence_scores"] = self.model_.confidence_scores
        ckpt["class_specific"] = self.model_.class_specific
        ckpt["param"] = self.model_.param
        ckpt["n_dim_"] = self.n_dim_
        if self.model_.extend_param:
            ckpt["param_ext"] = self.model_.param_ext

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
            The loaded model.
        
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
                            class_specific = False
                            )

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
        
        model.is_fitted = True

        moval_model.model_ = model
        moval_model.n_dim_ = loaded_ckpt["n_dim_"]

        return moval_model
    
# Dynamically remove the set_fit_request method
del MOVAL.set_fit_request