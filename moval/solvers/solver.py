import dataclasses as dataclasses
import abc
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
import math
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
            estim_perf_all = []
            kcls_all = []
            for kcls in range(self.batch * self.opt_kcls, np.min((self.batch * (self.opt_kcls + 1), self.model.num_class))):

                self.model.param[kcls] = x
                
                if self.metric == "accuracy":
                    if self.model.extend_param:
                        _, estim_perf = self.model.estimate_accuracy(self.inp, midstage = True, gt_guide = self.gt_guide)
                    else:
                        _, estim_perf = self.model.estimate_accuracy(self.inp, gt_guide = self.gt_guide)
                    estim_perf_all.append(estim_perf[kcls])
                elif self.metric == "sensitivity":
                    probability = self.model.calculate_probability(self.inp, midstage = True, appr=True)
                    estim_perf = self.model.estimate_sensitivity(self.inp, probability, gt_guide = self.gt_guide)
                    estim_perf_all.append(estim_perf[kcls])
                elif self.metric == "precision":
                    probability = self.model.calculate_probability(self.inp, midstage = True, appr=True, full=True)
                    estim_perf = self.model.estimate_precision(probability, gt_guide = self.gt_guide)
                    estim_perf_all.append(estim_perf[kcls])
                elif self.metric == "f1score":
                    probability = self.model.calculate_probability(self.inp, midstage = True, appr=True)
                    estim_perf = self.model.estimate_f1score(self.inp, probability, gt_guide = self.gt_guide)
                    estim_perf_all.append(estim_perf[kcls])
                elif self.metric == "auc":
                    probability = self.model.calculate_probability(self.inp, midstage = True, appr=True, full=True)
                    estim_perf = self.model.estimate_auc(probability, gt_guide = self.gt_guide, sel_cls = kcls)
                    estim_perf_all.append(estim_perf[0])
                else:
                    ValueError(f"Unsupported metric '{self.metric}'")
                kcls_all.append(kcls)

            err = self.criterions(self.inp, self.gt, estim_perf_all, kcls_all)

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
            estim_perf_all = []
            kcls_all = []
            for kcls in range(self.batch * self.opt_kcls, np.min((self.batch * (self.opt_kcls + 1), self.model.num_class))):
                self.model.param_ext[kcls] = x

                if self.metric == "accuracy":
                    _, estim_perf = self.model.estimate_accuracy(self.inp, gt_guide = self.gt_guide)
                    estim_perf_all.append(estim_perf[kcls])
                elif self.metric == "sensitivity":
                    probability = self.model.calculate_probability(self.inp, appr=True)
                    estim_perf = self.model.estimate_sensitivity(self.inp, probability, gt_guide = self.gt_guide)
                    estim_perf_all.append(estim_perf[kcls])
                elif self.metric == "precision":
                    probability = self.model.calculate_probability(self.inp, appr=True, full=True)
                    estim_perf = self.model.estimate_precision(probability, gt_guide = self.gt_guide)
                    estim_perf_all.append(estim_perf[kcls])
                elif self.metric == "f1score":
                    probability = self.model.calculate_probability(self.inp, appr=True)
                    estim_perf = self.model.estimate_f1score(self.inp, probability, gt_guide = self.gt_guide)
                    estim_perf_all.append(estim_perf[kcls])
                elif self.metric == "auc":
                    probability = self.model.calculate_probability(self.inp, appr=True, full=True)
                    estim_perf = self.model.estimate_auc(probability, gt_guide = self.gt_guide, sel_cls = self.kcls)
                    estim_perf_all.append(estim_perf[0])
                else:
                    ValueError(f"Unsupported metric '{self.metric}'")
                kcls_all.append(kcls)

            err = self.criterions(self.inp, self.gt, estim_perf_all, kcls_all)

        return err

    def kcls_order_list(self, inp: np.ndarray, exclusive_background: bool = False) -> List[Iterable]:
        """Generate a list of kcls, such that the predicted samples are in descending order.

        Args:
            inp: The network output (logits) of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            exclusive_background: If ``False``, we exclude the background class (the most majority class).
        
        Return:
            kcls_list: a list of class index. The first element should correspond to the most major class.

        """
        kcls_sample = np.zeros(self.opt_class)
        if self.model.mode == "classification":
            pred = np.argmax(inp, axis = 1)
            for opt_kcls in range(self.opt_class):
                for kcls in range(self.batch * opt_kcls, np.min((self.batch * (opt_kcls + 1), self.model.num_class))):
                    kcls_sample[opt_kcls] = kcls_sample[opt_kcls] + np.sum(pred == kcls)
        else:
            for kcls in range(self.model.num_class):
                for n_case in range(len(inp)):
                    pred_case = np.argmax(inp[n_case], axis = 0)
                    kcls_sample[kcls] = kcls_sample[kcls] + np.sum(pred_case == kcls)
        
        kcls_list = np.argsort(kcls_sample)
        if exclusive_background:
            kcls_list = kcls_list[:-1]

        return kcls_list[::-1]

    def fit(
        self,
        inp: Union[List[Iterable], np.ndarray],
        gt: Union[List[Iterable], np.ndarray],
        batch: int = 1
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Fit the model based on scipy backend.
        
        Args:
            inp: The network output (logits) of shape ``(n, d)`` for classification and a list of n ``(d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation of shape ``(n, )`` for classification and a list of n ``(H, W, (D))`` for segmentation.
            batch: To match the group-wise accuracy with group-wise confidence score. batch is the group size. This is useful when validation data per case is few.
        
        Note:
            ``batch`` is only used for class-specific variants. It does NOT support segmentation tasks as it is always unnecessary.
            
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
        self.batch = batch
        self.opt_class = math.ceil(self.model.num_class / batch)
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
            if self.model.estim_algorithm == "doc-model":
                # the range of doc estimation is [0, 2].
                initial_conditions = [np.array([0.01]), np.array([0.1]), np.array([0.5]), np.array([1.5]), np.array([1.8])]
            else:
                # the range of ts estimation is [0, inf].
                initial_conditions = [np.array([0.1]), np.array([0.5]), np.array([3.]), np.array([5.]), np.array([10.])]
            print(f"Opitimizing with {inp.shape[0]} samples...")
            
        else:
            if self.model.estim_algorithm == "doc-model":
                # the range of doc estimation is [0, 2].
                initial_conditions = [np.array([0.01]), np.array([1.5])]
            else:
                # the range of ts estimation is [0, inf].
                initial_conditions = [np.array([0.1]), np.array([5])]

            print(f"Opitimizing with {len(inp)} samples...")
            print("Be patient, it should take a while...")
            if self.metric != "accuracy":
                exclusive_background = True # leave the background class uncalibrated.

        # generate a list of length kcls, from high to low
        kcls_list = self.kcls_order_list(self.inp, exclusive_background)
        if self.metric == "auc" or self.metric == "precision":
            # these two largely depends on FP, threfore we optimize from the minority class
            kcls_list = kcls_list[::-1]

        if self.class_specific:
            
            if self.model.mode == "classification":
                pred = np.argmax(inp, axis = 1)
            else:
                preds = []
                for n_case in range(len(inp)):
                    preds.append(np.argmax(inp[n_case], axis = 0))

            for opt_kcls in kcls_list:
                self.opt_kcls = opt_kcls

                # do the class-wise optimization process, if 
                # a) there is any corresponding predictions, as it is possible for alignment, metric = 0.
                # and
                # b) there is any corresponding GT, as the alignment would has some meaning.
                flag_case_a = False
                flag_case_b = False
                if self.model.mode == "classification":
                    for kcls in range(self.batch * opt_kcls, np.min((self.batch * (opt_kcls + 1), self.model.num_class))):
                        if np.sum(pred == kcls) > 0:
                            flag_case_a = True
                        if np.sum(gt == kcls) > 0:
                            flag_case_b = True
                else:
                    for n_case in range(len(inp)):
                        if np.sum(preds[n_case] == opt_kcls) > 0:
                            flag_case_a = True
                        if np.sum(gt[n_case] == opt_kcls) > 0:
                            flag_case_b = True

                if flag_case_a and flag_case_b:

                    optimization_result = scipy.optimize.minimize(
                                    fun = self.eval_func,
                                    x0 = x0,
                                    method = optimization_method,
                                    bounds = [(1e-06,None)],
                                    tol = 1e-07)
                    
                    optimized_param = optimization_result.x[0]

                    if optimization_result.fun > search_threshold:
                        # change the initial state, if we are not satisfied with the optimization results.
                        print(f"Not satisfied with initial optimization results of param for class {opt_kcls}, trying more initial states...")
                        if self.model.estim_algorithm == "atc-model":
                            score = self.model.conf(inp)
                            if self.model.conf.normalization:
                                score = self.model._normalize(score)
                            score_ = []
                            if self.model.mode == "classification":
                                for kcls in range(self.batch * opt_kcls, np.min((self.batch * (opt_kcls + 1), self.model.num_class))):
                                    score_.append(score[np.argmax(inp, axis = 1) == kcls])
                                score_ = np.concatenate(score_)
                                initial_conditions = [np.array([np.percentile(score_, 20)]), 
                                                      np.array([np.percentile(score_, 40)]), 
                                                      np.array([np.percentile(score_, 50)]), 
                                                      np.array([np.percentile(score_, 60)]), 
                                                      np.array([np.percentile(score_, 80)])]
                            else:
                                for n_case in range(len(inp)):
                                    score_flatten = score[n_case].flatten() # ``n``
                                    pred_flatten = np.argmax(inp[n_case], axis = 0).flatten()
                                    score_.append(score_flatten[pred_flatten == opt_kcls])
                                score_ = np.concatenate(score_)
                                initial_conditions = [np.array([np.percentile(score_, 20)]), 
                                                      np.array([np.percentile(score_, 50)])]
                        results = []
                        results.append((optimization_result.fun, optimization_result.x[0]))
                        cnt_guess = 0
                        for initial_guess in initial_conditions:
                            optimization_result = scipy.optimize.minimize(
                                    fun = self.eval_func,
                                    x0 = initial_guess,
                                    method = optimization_method,
                                    bounds = [(1e-06,None)],
                                    tol = 1e-07)
                            results.append((optimization_result.fun, optimization_result.x[0]))
                            cnt_guess += 1
                            print(f"Tried {cnt_guess}/{len(initial_conditions)} times.")
                        
                        print(f"Starting from {initial_conditions}")
                        print(f"Optimization results are {results}")

                        optimized_param = min(results, key=lambda x: x[0])[1]

                    for kcls in range(self.batch * opt_kcls, np.min((self.batch * (opt_kcls + 1), self.model.num_class))):
                        self.model.param[kcls] = optimized_param
        else:
            optimization_result = scipy.optimize.minimize(
                            fun = self.eval_func,
                            x0 = x0,
                            method = optimization_method,
                            bounds = [(1e-06,None)],
                            tol = 1e-07)
        
            optimized_param = optimization_result.x

            if optimization_result.fun > search_threshold:
                # change the initial state, if we are not satisfied with the optimization results.
                print(f"Not satisfied with initial optimization results of param, trying more initial states...")
                if self.model.estim_algorithm == "atc-model":
                    score = self.model.conf(inp)
                    if self.model.conf.normalization:
                        score = self.model._normalize(score)
                    if self.model.mode == "classification":
                        score_ = score
                        initial_conditions = [np.array([np.percentile(score_, 20)]), 
                                                    np.array([np.percentile(score_, 40)]), 
                                                    np.array([np.percentile(score_, 50)]), 
                                                    np.array([np.percentile(score_, 60)]), 
                                                    np.array([np.percentile(score_, 80)])]
                    else:
                        score_ = []
                        for n_case in range(len(inp)):
                            score_flatten = score[n_case].flatten() # ``n``
                            score_.append(score_flatten)
                        score_ = np.concatenate(score_)
                        initial_conditions = [np.array([np.percentile(score_, 20)]), 
                                                    np.array([np.percentile(score_, 50)])]
                results = []
                results.append((optimization_result.fun, optimization_result.x))
                cnt_guess = 0
                for initial_guess in initial_conditions:
                    optimization_result = scipy.optimize.minimize(
                            fun = self.eval_func,
                            x0 = initial_guess,
                            method = optimization_method,
                            bounds = [(1e-06,None)],
                            tol = 1e-07)
                    results.append((optimization_result.fun, optimization_result.x))
                    cnt_guess += 1
                    print(f"Tried {cnt_guess}/{len(initial_conditions)} times.")
                
                print(f"Starting from {initial_conditions}")
                print(f"Optimization results are {results}")

                optimized_param = min(results, key=lambda x: x[0])[1]

            self.model.param = optimized_param

        if self.model.extend_param:

            if self.class_specific:
                for opt_kcls in kcls_list:
                    self.opt_kcls = opt_kcls

                    # do the class-wise optimization process, if 
                    # a) there is any corresponding predictions, as it is possible for alignment, metric = 0.
                    # and
                    # b) there is any corresponding GT, as the alignment would has some meaning.
                    flag_case_a = False
                    flag_case_b = False
                    if self.model.mode == "classification":
                        for kcls in range(self.batch * opt_kcls, np.min((self.batch * (opt_kcls + 1), self.model.num_class))):
                            if np.sum(pred == kcls) > 0:
                                flag_case_a = True
                            if np.sum(gt == kcls) > 0:
                                flag_case_b = True
                    else:
                        for n_case in range(len(inp)):
                            if np.sum(preds[n_case] == opt_kcls) > 0:
                                flag_case_a = True
                            if np.sum(gt[n_case] == opt_kcls) > 0:
                                flag_case_b = True

                    if flag_case_a and flag_case_b:

                        optimization_result = scipy.optimize.minimize(
                                        fun = self.eval_func_ext,
                                        x0 = x0,
                                        method = optimization_method,
                                        bounds = [(1e-03,None)],
                                        tol = 1e-07)
                        
                        optimized_param_ext = optimization_result.x[0]

                        if optimization_result.fun > search_threshold:
                            # change the initial state, if we are not satisfied with the optimization results.
                            print(f"Not satisfied with initial optimization results of param_ext for class {opt_kcls}, trying more initial states...")
                            # change atc to be consistent with the range of confidence score
                            # get the max and min value here.
                            score = self.model.calibrate(inp, midstage = True)
                            score_ = []
                            if self.model.mode == "classification":
                                for kcls in range(self.batch * opt_kcls, np.min((self.batch * (opt_kcls + 1), self.model.num_class))):
                                    score_.append(score[np.argmax(inp, axis = 1) == kcls])
                                score_ = np.concatenate(score_)
                                initial_conditions_atc = [np.array([np.percentile(score_, 20)]), 
                                                            np.array([np.percentile(score_, 40)]), 
                                                            np.array([np.percentile(score_, 50)]), 
                                                            np.array([np.percentile(score_, 60)]), 
                                                            np.array([np.percentile(score_, 80)])]
                            else:
                                for n_case in range(len(inp)):
                                    score_flatten = score[n_case].flatten() # ``n``
                                    pred_flatten = np.argmax(inp[n_case], axis = 0).flatten()
                                    score_.append(score_flatten[pred_flatten == opt_kcls])
                                score_ = np.concatenate(score_)
                                initial_conditions_atc = [np.array([np.percentile(score_, 20)]), 
                                                            np.array([np.percentile(score_, 50)])]
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
                            
                            print(f"Starting from {initial_conditions_atc}")
                            print(f"Optimization results are {results}")

                            optimized_param_ext = min(results, key=lambda x: x[0])[1]

                        for kcls in range(self.batch * opt_kcls, np.min((self.batch * (opt_kcls + 1), self.model.num_class))):
                            self.model.param_ext[kcls] = optimized_param_ext

            else:
                optimization_result = scipy.optimize.minimize(
                                fun = self.eval_func_ext,
                                x0 = x0,
                                method = optimization_method,
                                bounds = [(1e-03,None)],
                                tol = 1e-07)

                optimized_param_ext = optimization_result.x

                if optimization_result.fun > search_threshold:
                    # change the initial state, if we are not satisfied with the optimization results.
                    print(f"Not satisfied with initial optimization results of param_ext, trying more initial states...")
                    # change atc to be consistent with the range of confidence score
                    # get the max and min value here.
                    score = self.model.calibrate(inp, midstage = True)
                    if self.model.mode == "classification":
                        score_ = score
                        initial_conditions_atc = [np.array([np.percentile(score_, 20)]), 
                                                    np.array([np.percentile(score_, 40)]), 
                                                    np.array([np.percentile(score_, 50)]), 
                                                    np.array([np.percentile(score_, 60)]), 
                                                    np.array([np.percentile(score_, 80)])]
                    else:
                        score_ = []
                        for n_case in range(len(inp)):
                            score_flatten = score[n_case].flatten() # ``n``
                            score_.append(score_flatten)
                        score_ = np.concatenate(score_)
                        initial_conditions_atc = [np.array([np.percentile(score_, 20)]), 
                                                    np.array([np.percentile(score_, 50)])]
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
                    
                    print(f"Starting from {initial_conditions_atc}")
                    print(f"Optimization results are {results}")

                    optimized_param_ext = min(results, key=lambda x: x[0])[1]

                self.model.param_ext = optimized_param_ext

        # I need save the normalization parameter, if the confidence needs to be normalized.
        _ = self.model.calibrate(self.inp)

        self.model.eval()
        self.model.is_fitted = True

        if self.model.extend_param:
            return self.model.param, self.model.param_ext
        else:
            return self.model.param
