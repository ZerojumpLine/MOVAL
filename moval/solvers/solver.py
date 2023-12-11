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
        class_specific (bool):
            If ``True``, the calculation will match class-wise confidence to class-wise accuracy/DSC.
        criterions (moval.solver.Calibrate):
            Calibration criterions. This will be utilized by solvers.

    """

    model: moval.models.Model
    class_specific: bool = dataclasses.field(init=False)
    
    def __post_init__(self):
        self.class_specific = self.model.class_specific

        if self.model.mode == "classification":
            self.criterions = moval.solvers.clsCalibrate(class_specific=self.class_specific)
        elif self.model.mode == "segmentation":
            self.criterions = moval.solvers.segCalibrate(class_specific=self.class_specific)
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
                estim_acc, estim_cls = self.model(self.inp, midstage = True, gt = self.gt_guide)
            else:
                estim_acc, estim_cls = self.model(self.inp, gt = self.gt_guide)
            err = self.criterions(self.inp, self.gt, estim_acc)
        else:
            self.model.param[self.kcls] = x
            if self.model.extend_param:
                estim_acc, estim_cls = self.model(self.inp, midstage = True, gt = self.gt_guide)
            else:
                estim_acc, estim_cls = self.model(self.inp, gt = self.gt_guide)
            err = self.criterions(self.inp, self.gt, estim_cls)
            err = np.mean(err)

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
            estim_acc, estim_cls = self.model(self.inp, gt = self.gt_guide)
            err = self.criterions(self.inp, self.gt, estim_acc)
        else:
            self.model.param_ext[self.kcls] = x
            estim_acc, estim_cls = self.model(self.inp, gt = self.gt_guide)
            err_cls = self.criterions(self.inp, self.gt, estim_cls)
            err = err_cls[self.kcls]

        return err

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

        print(f"Starting optimizing for model {self.model.estim_algorithm} with confidence {self.model.confidence_scores}, class specific is {self.model.class_specific}.")

        if self.model.estim_algorithm == "ac-model":
            # do not need optimization, end task as soon as we possible!

            self.model.train()

            if self.model.conf.normalization:
                _, _ = self.model(inp)

            self.model.eval()
            self.model.is_fitted = True

            return self.model.param

        self.inp = inp
        self.gt = gt
        if self.model.mode == "segmentation":
            self.gt_guide = self.gt
        else:
            self.gt_guide = None

        self.model.train()

        optimization_method = 'Nelder-Mead'
        x0 = np.array([1.0])
        search_threshold = 0.01 # if the optimized results are larger than this, we go through the initial conditions

        if self.model.mode == "classification":
            initial_conditions = [np.array([0.001]), np.array([0.1]), np.array([0.2]), np.array([0.3]), np.array([0.4]),
                                  np.array([0.5]), np.array([0.6]), np.array([0.7]), np.array([0.8]), np.array([0.9])]
            print(f"Opitimizing with {inp.shape[0]} samples...")
        else:
            initial_conditions = [np.array([0.001]), np.array([0.5])]
            print(f"Opitimizing with {len(inp)} samples...")
            print("Be patient, it should take a while...")

        if self.class_specific:
            for kcls in range(self.model.num_class):
                self.kcls = kcls
                optimization_result = scipy.optimize.minimize(
                                fun = self.eval_func,
                                x0 = x0,
                                method = optimization_method,
                                bounds = [(1e-06,None)],
                                tol = 1e-07)
                
                optimized_param = optimization_result.x[0]

                if optimization_result.fun > search_threshold:
                    # change the initial state, if we are not satisfied with the optimization results.
                    print(f"Not satisfied with initial optimization results of param, trying more initial states...")
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
                    
                    optimized_param = min(results, key=lambda x: x[0])[1]

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
                
                optimized_param = min(results, key=lambda x: x[0])[1]

            self.model.param = optimized_param

        if self.model.extend_param:
            # If the model needs normalization, it learns ``min_value`` during the first stage
            # It is maybe not desrible for the second stage, sometimes hurt performance.
            # Therefore, here we decards the learned parameters.
            if self.model.conf.normalization:
                if self.model.class_specific:
                    self.model.param = np.ones(self.model.num_class)
                else:
                    self.model.param = np.array([1])

            if self.class_specific:
                for kcls in range(self.model.num_class):
                    self.kcls = kcls
                    optimization_result = scipy.optimize.minimize(
                                    fun = self.eval_func_ext,
                                    x0 = x0,
                                    method = optimization_method,
                                    bounds = [(1e-03,None)],
                                    tol = 1e-07)
                    
                    optimized_param = optimization_result.x[0]

                    if optimization_result.fun > search_threshold:
                        # change the initial state, if we are not satisfied with the optimization results.
                        print(f"Not satisfied with initial optimization results of param_ext, trying more initial states...")
                        results = []
                        results.append((optimization_result.fun, optimization_result.x[0]))
                        cnt_guess = 0
                        for initial_guess in initial_conditions:
                            optimization_result = scipy.optimize.minimize(
                                    fun = self.eval_func_ext,
                                    x0 = initial_guess,
                                    method = optimization_method,
                                    bounds = [(1e-03,None)],
                                    tol = 1e-07)
                            results.append((optimization_result.fun, optimization_result.x[0]))
                            cnt_guess += 1
                            print(f"Tried {cnt_guess}/{len(initial_conditions)} times.")
                        
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
                    for initial_guess in initial_conditions:
                        optimization_result = scipy.optimize.minimize(
                                fun = self.eval_func_ext,
                                x0 = initial_guess,
                                method = optimization_method,
                                bounds = [(1e-03,None)],
                                tol = 1e-07)
                        results.append((optimization_result.fun, optimization_result.x))
                        cnt_guess += 1
                        print(f"Tried {cnt_guess}/{len(initial_conditions)} times.")
                    
                    optimized_param = min(results, key=lambda x: x[0])[1]

                self.model.param_ext = optimized_param

            self.model.eval()
            self.model.is_fitted = True

            return self.model.param, self.model.param_ext

        self.model.eval()
        self.model.is_fitted = True

        return self.model.param

