import pytest
import numpy as np
import itertools

import moval
import moval.models
import moval.solvers
import moval.registry

"""Test the confidence calculation functions.

    Examples:
        >>> pytest tests/4_test_solver.py

"""

def test_registry():
    assert moval.registry.is_registry(moval.solvers)
    assert moval.registry.is_registry(moval.solvers, check_docs=True)

# estim_algorithm = "ts-model"
# mode = "classification"
# metric = "f1score"
# numclass = 3
# confidence_scores = "max_class_probability-conf"
# class_specific = True

@pytest.mark.parametrize(
        "estim_algorithm, mode, metric, numclass, confidence_scores, class_specific", 
        list(itertools.product(moval.models.get_estim_options(),
                               ["classification", "segmentation"],
                               ["accuracy", "sensitivity", "precision", "f1score", "auc"],
                               [3], 
                               moval.models.get_conf_options(),
                               [False, True])),
)
def test_solver(estim_algorithm, mode, metric, numclass, confidence_scores, class_specific):

    model = moval.models.init(
            estim_algorithm,
            mode = mode,
            num_class = numclass,
            confidence_scores = confidence_scores,
            class_specific = class_specific
            )
    model.train()
    solver = moval.solvers.init("base-solver", model = model, metric = metric)
    assert isinstance(solver, moval.solvers.Solver)

    if mode == "classification":
        inp = np.random.randn(100, numclass) # 100 samples, 3 classes
        gt = np.random.randint(0, numclass, (100, ))
        _estim_acc, _estim_acc_allcls = model.estimate_accuracy(inp)
        #
        solver.fit(inp, gt)
        assert model.is_fitted == True

        if metric == "accuracy":

            estim_acc, estim_acc_allcls = model.estimate_accuracy(inp)

            criterions = moval.solvers.clsCalibrate(class_specific=class_specific, metric=metric)

            if not class_specific:
                _err = criterions(inp, gt, _estim_acc)
                err = criterions(inp, gt, estim_acc)
                if estim_algorithm != "ac-model":
                    assert err <= _err
            else:
                _err = criterions(inp, gt, _estim_acc_allcls)
                err = criterions(inp, gt, estim_acc_allcls)
                if estim_algorithm != "ac-model":
                    assert (err <= _err).any()
    
    if mode == "segmentation":
        # segmentation logit input
        inp = []
        gt = []
        for _ in range(5):
            inp.append(np.random.randn(numclass, 10, 10, 10)) # 5 samples, 3 classes, of volume shape ``(10, 10, 10)``
            gt.append(np.random.randint(0, numclass, (10, 10, 10)))

        _estim_acc, _estim_acc_allcls =  model.estimate_accuracy(inp)
        #
        solver.fit(inp, gt)
        assert model.is_fitted == True

        if metric == "accuracy":

            estim_acc, estim_acc_allcls =  model.estimate_accuracy(inp)
            
            criterions = moval.solvers.segCalibrate(class_specific=class_specific, metric=metric)

            if not class_specific:
                _err = criterions(inp, gt, _estim_acc)
                err = criterions(inp, gt, estim_acc)
                if estim_algorithm != "ac-model":
                    assert err <= _err
            else:
                _err = criterions(inp, gt, _estim_acc_allcls)
                err = criterions(inp, gt, estim_acc_allcls)
                if estim_algorithm != "ac-model":
                    assert (err <= _err).any()