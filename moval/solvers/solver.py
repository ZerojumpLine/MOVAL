import dataclasses as dataclasses
import abc
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
import scipy
import moval.models
import moval.solvers
from moval.solvers import register

@register("base-solver")
@dataclasses.dataclass
class Solver(abc.ABC):
    """Solver base class.

    A solver contains helper methods for bundling a model, criterion and optimizer.
    This solver will save the results in :py:attr:`models.param`.

    Atrributes:
        model (moval.models.Model): 
            The model to calibrate the network outputs.
        metric (str):
            The considered performance metric to align.
        class_specific (bool):
            If ``True``, the calculation will match class-wise confidence to class-wise accuracy/DSC.
        criterions (moval.solver.Calibrate):
            Calibration criterions. This will be utilized by solvers.

    """

    model: moval.models.Model
    metric: str
    class_specific: bool = dataclasses.field(init=False)
    
    def __post_init__(self):
        self.class_specific = self.model.class_specific

        if self.model.mode == "classification":
            self.criterions = moval.solvers.clsCalibrate(class_specific = self.class_specific, metric = self.metric)
        elif self.model.mode == "segmentation":
            self.criterions = moval.solvers.segCalibrate(class_specific = self.class_specific, metric = self.metric)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def eval_func(self, x: float) -> float:
        """The evaluation function for scipy optimization.
        
        Args:
            x: The variable (numpy.float64) to be optimized.
        
        Return:
            err: The caculated err taking x as input.

        """

        if not self.class_specific:
            self.model.param = x
            if self.model.extend_param:
                estim_acc, _ = self.model.estimate_accuracy(self.inp, midstage = True, gt_guide = self.gt_guide)
            else:
                estim_acc, _ = self.model.estimate_accuracy(self.inp, gt_guide = self.gt_guide)
            err = self.criterions(self.inp, self.gt, estim_acc)
        else:
            self.model.param[self.kcls] = x
            
            if self.metric == "accuracy":
                if self.model.extend_param:
                    _, estim_perf = self.model.estimate_accuracy(self.inp, midstage = True, gt_guide = self.gt_guide)
                else:
                    _, estim_perf = self.model.estimate_accuracy(self.inp, gt_guide = self.gt_guide)
            elif self.metric == "sensitivity":
                probability = self.model.calculate_probability(self.inp, midstage = True, appr=True)
                estim_perf = self.model.estimate_sensitivity(self.inp, probability, gt_guide = self.gt_guide)
            elif self.metric == "precision":
                probability = self.model.calculate_probability(self.inp, midstage = True, appr=True, full=True)
                estim_perf = self.model.estimate_precision(probability, gt_guide = self.gt_guide)
            elif self.metric == "f1score":
                probability = self.model.calculate_probability(self.inp, midstage = True, appr=True)
                estim_perf = self.model.estimate_f1score(self.inp, probability, gt_guide = self.gt_guide)
            elif self.metric == "auc":
                probability = self.model.calculate_probability(self.inp, midstage = True, appr=True, full=True)
                estim_perf = self.model.estimate_auc(probability, gt_guide = self.gt_guide, sel_cls = self.kcls)
            else:
                ValueError(f"Unsupported metric '{self.metric}'")

            err = self.criterions(self.inp, self.gt, estim_perf, self.kcls)

        return err

    def eval_func_ext(self, x: float) -> float:
        """The evaluation function for scipy optimization of extended parameters.
        
        Args:
            x: The variable to be optimized.
        
        Return:
            err: The caculated err taking x as input.

        """
        if not self.class_specific:
            self.model.param_ext = x
            estim_acc, _ = self.model.estimate_accuracy(self.inp, gt_guide = self.gt_guide)
            err = self.criterions(self.inp, self.gt, estim_acc)
        else:
            self.model.param_ext[self.kcls] = x

            if self.metric == "accuracy":
                _, estim_perf = self.model.estimate_accuracy(self.inp, gt_guide = self.gt_guide)
            elif self.metric == "sensitivity":
                probability = self.model.calculate_probability(self.inp, appr=True)
                estim_perf = self.model.estimate_sensitivity(self.inp, probability, gt_guide = self.gt_guide)
            elif self.metric == "precision":
                probability = self.model.calculate_probability(self.inp, appr=True, full=True)
                estim_perf = self.model.estimate_precision(probability, gt_guide = self.gt_guide)
            elif self.metric == "f1score":
                probability = self.model.calculate_probability(self.inp, appr=True)
                estim_perf = self.model.estimate_f1score(self.inp, probability, gt_guide = self.gt_guide)
            elif self.metric == "auc":
                probability = self.model.calculate_probability(self.inp, appr=True, full=True)
                estim_perf = self.model.estimate_auc(probability, gt_guide = self.gt_guide, sel_cls = self.kcls)
            else:
                ValueError(f"Unsupported metric '{self.metric}'")

            err = self.criterions(self.inp, self.gt, estim_perf, self.kcls)

        return err

    def kcls_order_list(self, inp: np.ndarray, exclusive_background: bool = False) -> List[Iterable]:
        """Generate a list of kcls, such that the predicted samples are in ascending order.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            exclusive_background: If ``False``, we exclude the background class (the most majority class).
        
        Return:
            kcls_list: a list of class index. The first element should correspond to the most minority class.

        """
        numclass = self.model.num_class
        kcls_sample = np.zeros(numclass)
        if self.model.mode == "classification":
            pred = np.argmax(inp, axis = 1)
            for kcls in range(numclass):
                kcls_sample[kcls] = kcls_sample[kcls] + np.sum(pred == kcls)
        else:
            for kcls in range(numclass):
                for n_case in range(len(inp)):
                    pred_case = np.argmax(inp[n_case], axis = 0)
                    kcls_sample[kcls] = kcls_sample[kcls] + np.sum(pred_case == kcls)
        
        kcls_list = np.argsort(kcls_sample)
        if exclusive_background:
            kcls_list = kcls_list[:-1]

        return kcls_list

    def fit(
        self,
        inp: Union[List[Iterable], np.ndarray],
        gt: Union[List[Iterable], np.ndarray]
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Fit the model based on scipy backend.
        
        Args:
            inp: The network output (logits) of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation of shape ``(n, )`` for classification and a list of n ``(H, W, (D))`` for segmentation. 
        
        Return:
            param: The optimized parameters, of shape ``(d, )`` or a float.
            If the model contains extended parameters, will return Tuple[param, param_ext].

        """

        print(f"Starting optimizing for model {self.model.estim_algorithm} with confidence {self.model.confidence_scores} based on metric {self.metric}, class specific is {self.model.class_specific}.")

        if self.model.estim_algorithm == "ac-model":
            # do not need optimization, end task as soon as we possible!

            self.model.train()

            if self.model.conf.normalization:
                _, _ = self.model.estimate_accuracy(inp)

            self.model.eval()
            self.model.is_fitted = True

            return self.model.param

        self.inp = inp
        self.gt = gt
        # generate gt_guide from gt here.
        if self.model.mode == "segmentation":
            gt_guide = []
            for n_case in range(len(gt)):
                gt_case     = gt[n_case]
                gt_exist = []
                for k_cls in range(self.model.num_class):
                    gt_exist.append(np.sum(gt_case == k_cls) > 0)
                gt_guide.append(gt_exist)
            gt_guide = np.array(gt_guide)
            self.gt_guide = gt_guide
        else:
            self.gt_guide = None

        self.model.train()
        
        # some hyper parameters.
        optimization_method = 'Nelder-Mead'
        x0 = np.array([1.0])
        search_threshold = 0.01 # if the optimized results are larger than this, we go through the initial conditions
        exclusive_background = False

        if self.model.mode == "classification":
            initial_conditions_atc = [np.array([0.01]), np.array([0.1]), np.array([0.3]), np.array([0.5]), np.array([0.9])]
            if self.model.estim_algorithm == "atc-model":
                # the range of atc estimation is [0, 1].
                initial_conditions = [np.array([0.01]), np.array([0.1]), np.array([0.3]), np.array([0.5]), np.array([0.9])]
            elif self.model.estim_algorithm == "doc-model":
                # the range of doc estimation is [0, 2].
                initial_conditions = [np.array([0.01]), np.array([0.1]), np.array([0.5]), np.array([1.5]), np.array([1.8])]
            else:
                # the range of ts estimation is [0, inf].
                initial_conditions = [np.array([0.01]), np.array([0.5]), np.array([3.]), np.array([5.]), np.array([10.])]
            print(f"Opitimizing with {inp.shape[0]} samples...")
            
        else:
            initial_conditions_atc = [np.array([0.01]), np.array([0.5])]
            if self.model.estim_algorithm == "atc-model":
                # the range of atc estimation is [0, 1].
                initial_conditions = [np.array([0.01]), np.array([0.5])]
            elif self.model.estim_algorithm == "doc-model":
                # the range of doc estimation is [0, 2].
                initial_conditions = [np.array([0.01]), np.array([1.5])]
            else:
                # the range of ts estimation is [0, inf].
                initial_conditions = [np.array([0.01]), np.array([5])]

            print(f"Opitimizing with {len(inp)} samples...")
            print("Be patient, it should take a while...")
            if self.metric != "accuracy":
                exclusive_background = True # leave the background class uncalibrated.

        # generate a list of length kcls, from high to low
        kcls_list = self.kcls_order_list(self.inp, exclusive_background)
        if self.metric == "f1score":
            kcls_list = kcls_list[::-1]

        if self.class_specific:
            pred = np.argmax(inp, axis = 1)

            for kcls in kcls_list:
                self.kcls = kcls

                # skip the class-wise optimization process, if 
                # a) there isn't any corresponding predictions, as it is not possible for alignment, metric = 0.
                # b) there isn't any corresponding GT, as the alignment would has not much meaning.
                gt_pos_cls = np.where(gt == kcls)[0]
                pred_pos_cls = np.where(pred == kcls)[0]

                if len(gt_pos_cls) > 0 and len(pred_pos_cls) > 0:

                    optimization_result = scipy.optimize.minimize(
                                    fun = self.eval_func,
                                    x0 = x0,
                                    method = optimization_method,
                                    bounds = [(1e-03,None)],
                                    tol = 1e-07)
                    
                    optimized_param = optimization_result.x[0]

                    if optimization_result.fun > search_threshold:
                        # change the initial state, if we are not satisfied with the optimization results.
                        print(f"Not satisfied with initial optimization results of param for class {kcls}, trying more initial states...")
                        results = []
                        results.append((optimization_result.fun, optimization_result.x[0]))
                        cnt_guess = 0
                        for initial_guess in initial_conditions:
                            optimization_result = scipy.optimize.minimize(
                                    fun = self.eval_func,
                                    x0 = initial_guess,
                                    method = optimization_method,
                                    bounds = [(1e-03,None)],
                                    tol = 1e-07)
                            results.append((optimization_result.fun, optimization_result.x[0]))
                            cnt_guess += 1
                            print(f"Tried {cnt_guess}/{len(initial_conditions)} times.")
                        
                        optimized_param = min(results, key=lambda x: x[0])[1]

                    self.model.param[kcls] = optimized_param
        else:
            optimization_result = scipy.optimize.minimize(
                            fun = self.eval_func,
                            x0 = x0,
                            method = optimization_method,
                            bounds = [(1e-03,None)],
                            tol = 1e-07)
        
            optimized_param = optimization_result.x

            if optimization_result.fun > search_threshold:
                # change the initial state, if we are not satisfied with the optimization results.
                print(f"Not satisfied with initial optimization results of param, trying more initial states...")
                results = []
                results.append((optimization_result.fun, optimization_result.x))
                cnt_guess = 0
                for initial_guess in initial_conditions:
                    optimization_result = scipy.optimize.minimize(
                            fun = self.eval_func,
                            x0 = initial_guess,
                            method = optimization_method,
                            bounds = [(1e-03,None)],
                            tol = 1e-07)
                    results.append((optimization_result.fun, optimization_result.x))
                    cnt_guess += 1
                    print(f"Tried {cnt_guess}/{len(initial_conditions)} times.")
                
                optimized_param = min(results, key=lambda x: x[0])[1]

            self.model.param = optimized_param

        if self.model.extend_param:

            if self.class_specific:
                for kcls in kcls_list:
                    self.kcls = kcls

                    # skip the class-wise optimization process, if 
                    # a) there isn't any corresponding predictions, as it is not possible for alignment, metric = 0.
                    # b) there isn't any corresponding GT, as the alignment would has not much meaning.
                    gt_pos_cls = np.where(gt == kcls)[0]
                    pred_pos_cls = np.where(pred == kcls)[0]

                    if len(gt_pos_cls) > 0 and len(pred_pos_cls) > 0:

                        optimization_result = scipy.optimize.minimize(
                                        fun = self.eval_func_ext,
                                        x0 = x0,
                                        method = optimization_method,
                                        bounds = [(1e-03,None)],
                                        tol = 1e-07)
                        
                        optimized_param = optimization_result.x[0]

                        if optimization_result.fun > search_threshold:
                            # change the initial state, if we are not satisfied with the optimization results.
                            print(f"Not satisfied with initial optimization results of param for class {kcls}, trying more initial states...")
                            results = []
                            results.append((optimization_result.fun, optimization_result.x[0]))
                            cnt_guess = 0
                            for initial_guess in initial_conditions_atc:
                                optimization_result = scipy.optimize.minimize(
                                        fun = self.eval_func_ext,
                                        x0 = initial_guess,
                                        method = optimization_method,
                                        bounds = [(1e-03,None)],
                                        tol = 1e-07)
                                results.append((optimization_result.fun, optimization_result.x[0]))
                                cnt_guess += 1
                                print(f"Tried {cnt_guess}/{len(initial_conditions_atc)} times.")
                            
                            optimized_param = min(results, key=lambda x: x[0])[1]

                    self.model.param_ext[kcls] = optimized_param

            else:
                optimization_result = scipy.optimize.minimize(
                                fun = self.eval_func_ext,
                                x0 = x0,
                                method = optimization_method,
                                bounds = [(1e-03,None)],
                                tol = 1e-07)

                optimized_param = optimization_result.x

                if optimization_result.fun > search_threshold:
                    # change the initial state, if we are not satisfied with the optimization results.
                    print(f"Not satisfied with initial optimization results of param_ext, trying more initial states...")
                    results = []
                    results.append((optimization_result.fun, optimization_result.x))
                    cnt_guess = 0
                    for initial_guess in initial_conditions_atc:
                        optimization_result = scipy.optimize.minimize(
                                fun = self.eval_func_ext,
                                x0 = initial_guess,
                                method = optimization_method,
                                bounds = [(1e-03,None)],
                                tol = 1e-07)
                        results.append((optimization_result.fun, optimization_result.x))
                        cnt_guess += 1
                        print(f"Tried {cnt_guess}/{len(initial_conditions_atc)} times.")
                    
                    optimized_param = min(results, key=lambda x: x[0])[1]

                self.model.param_ext = optimized_param

        # I need save the normalization parameter, if the confidence needs to be normalized.
        _ = self.model.calibrate(self.inp)

        self.model.eval()
        self.model.is_fitted = True

        if self.model.extend_param:
            return self.model.param, self.model.param_ext
        else:
            return self.model.param
