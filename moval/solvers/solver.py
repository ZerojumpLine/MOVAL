import dataclasses as dataclasses
import abc
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

    def eval_func(self, x) -> float:
        """The evaluation function for scipy optimization.
        
        Args:
            x: The variable (numpy.float64) to be optimized.
        
        Return:
            err: The caculated err taking x as input.

        """
        if not self.class_specific:
            self.model.param = x
            estim_acc, estim_cls = self.model(self.inp)
            err = self.criterions(self.inp, self.gt, estim_acc)
        else:
            self.model.param[self.kcls] = x
            estim_acc, estim_cls = self.model(self.inp)
            err = self.criterions(self.inp, self.gt, estim_cls)
            err = np.mean(err)

        return err
    
    def eval_func_ext(self, x) -> float:
        """The evaluation function for scipy optimization of extended parameters.
        
        Args:
            x: The variable to be optimized.
        
        Return:
            err: The caculated err taking x as input.

        """
        if not self.class_specific:
            self.model.param_ext = x
            estim_acc, estim_cls = self.model(self.inp)
            err = self.criterions(self.inp, self.gt, estim_acc)
        else:
            self.model.param_ext[self.kcls] = x
            estim_acc, estim_cls = self.model(self.inp)
            err = self.criterions(self.inp, self.gt, estim_cls)
            err = np.mean(err)

        return err

    def fit(
        self,
        inp: np.ndarray,
        gt: np.ndarray
    ):
        """Fit the model based on scipy backend.
        
        Args:
            inp: The network output (logits) of shape ``(n, d)`` for classification and ``(n, d, H, W, (D))`` for segmentation. 
            gt: The cooresponding annotation of shape ``(n, )`` for classification and ``(n, H, W, (D))`` for segmentation
        
        Return:
            param: The optimized parameters, of shape ``(d, )`` or a float.
            If the model contains extended parameters, will return Tuple[param, param_ext].

        """
        self.inp = inp
        self.gt = gt

        self.model.train()

        if self.class_specific:
            for kcls in range(self.inp.shape[1]):
                self.kcls = kcls
                optimization_result = scipy.optimize.minimize(
                                fun = self.eval_func,
                                x0 = np.array([1.0]),
                                method = 'Nelder-Mead',
                                bounds = [(0,None)],
                                tol = 1e-07)
                self.model.param[kcls] = optimization_result.x[0]
        else:
            optimization_result = scipy.optimize.minimize(
                            fun = self.eval_func,
                            x0 = np.array([1.0]),
                            method = 'Nelder-Mead',
                            bounds = [(0,None)],
                            tol = 1e-07)
        
            self.model.param = optimization_result.x

        if self.model.extend_param:
            if self.class_specific:
                for kcls in range(self.inp.shape[1]):
                    self.kcls = kcls
                    optimization_result = scipy.optimize.minimize(
                                    fun = self.eval_func_ext,
                                    x0 = np.array([1.0]),
                                    method = 'Nelder-Mead',
                                    bounds = [(0,None)],
                                    tol = 1e-07)
                    self.model.param_ext[kcls] = optimization_result.x[0]
            else:
                optimization_result = scipy.optimize.minimize(
                                fun = self.eval_func_ext,
                                x0 = np.array([1.0]),
                                method = 'Nelder-Mead',
                                bounds = [(0,None)],
                                tol = 1e-07)
            
                self.model.param_ext = optimization_result.x
            self.model.eval()
            self.model.is_fitted = True

            return self.model.param, self.model.param_ext

        self.model.eval()
        self.model.is_fitted = True

        return self.model.param

