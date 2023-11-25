import pytest
import numpy as np
import itertools

import moval
import moval.models
import moval.registry

"""Test the confidence calculation functions.

    Examples:
        >>> pytest tests/test_models.py

"""

def test_registry():
    assert moval.registry.is_registry(moval.models)
    assert moval.registry.is_registry(moval.models, check_docs=True)

@pytest.mark.parametrize(
        "estim_algorithm, mode, numclass, confidence_scores, class_specific", 
        list(itertools.product(moval.models.get_estim_options(),
                               ["classification", "segmentation"],
                               [2, 10],
                               moval.models.get_conf_options(),
                               [False, True])),
)
def test_model(estim_algorithm, mode, numclass, confidence_scores, class_specific):
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
        estim_acc, estim_acc_allcls = model(inp)
        assert estim_acc <= 1 and estim_acc >= 0
        assert len(estim_acc_allcls) == numclass
        assert np.max(estim_acc) <= 1 and np.min(estim_acc) >= 0
    
    if mode == "segmentation":
        # segmentation logit input
        inp = []
        for _ in range(5):
            inp.append(np.random.randn(numclass, 50, 50, 50)) # 10 samples, 3 classes, of volume shape ``(100, 100, 100)``
        estim_acc, estim_dsc = model(inp)
        assert estim_acc <= 1 and estim_acc >= 0
        assert len(estim_dsc) == numclass
        assert np.max(estim_dsc) <= 1 and np.min(estim_dsc) >= 0