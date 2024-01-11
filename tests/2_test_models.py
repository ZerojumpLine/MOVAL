import pytest
import numpy as np
import itertools

import moval
import moval.models
import moval.registry

"""Test the confidence calculation functions.

    Examples:
        >>> pytest tests/2_test_models.py

"""

def test_registry():
    assert moval.registry.is_registry(moval.models)
    assert moval.registry.is_registry(moval.models, check_docs=True)

# estim_algorithm = "ts-model"
# mode = "segmentation"
# numclass = 2
# confidence_scores = "max_class_probability-conf"
# class_specific = True

@pytest.mark.parametrize(
        "estim_algorithm, mode, metric, numclass, confidence_scores, class_specific", 
        list(itertools.product(moval.models.get_estim_options(),
                               ["classification", "segmentation"],
                               ["accuracy", "sensitivity", "precision", "f1score", "auc"],
                               [2, 10],
                               moval.models.get_conf_options(),
                               [False, True])),
)
def test_model(estim_algorithm, mode, metric, numclass, confidence_scores, class_specific):
    model = moval.models.init(
            estim_algorithm,
            mode = mode,
            num_class = numclass,
            confidence_scores = confidence_scores,
            class_specific = class_specific
            )
    assert isinstance(model, moval.models.Model)

    model.train()
    if mode == "classification":
        # classification logit input
        inp = np.random.randn(100, numclass) # 100 samples
    else:
        inp = []
        for _ in range(5):
            inp.append(np.random.randn(numclass, 20, 20, 20)) # 10 samples, 2 / 10 classes, of volume shape ``(20, 20, 20)``

    if metric == "accuracy":
        estim_acc, estim_acc_allcls = model.estimate_accuracy(inp)
        assert estim_acc <= 1 and estim_acc >= 0
        assert len(estim_acc_allcls) == numclass
        assert np.max(estim_acc_allcls) <= 1 and np.min(estim_acc_allcls) >= 0
    elif metric == "sensitivity":
        probability = model.calculate_probability(inp)
        estim_sensitivity = model.estimate_sensitivity(inp, probability)
        assert len(estim_sensitivity) == numclass
        assert np.max(estim_sensitivity) <= 1 and np.min(estim_sensitivity) >= 0
    elif metric == "precision":
        probability = model.calculate_probability(inp)
        estim_precision = model.estimate_precision(probability)
        assert len(estim_precision) == numclass
        assert np.max(estim_precision) <= 1 and np.min(estim_precision) >= 0
    elif metric == "f1score":
        probability = model.calculate_probability(inp)
        estim_f1score = model.estimate_f1score(inp, probability)
        assert len(estim_f1score) == numclass
        assert np.max(estim_f1score) <= 1 and np.min(estim_f1score) >= 0
    else:
        probability = model.calculate_probability(inp)
        estim_auc = model.estimate_auc(probability)
        assert len(estim_auc) == numclass
        assert np.max(estim_auc) <= 1 and np.min(estim_auc) >= 0
    