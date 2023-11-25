import pytest
import numpy as np

import moval
import moval.solvers

"""Test the confidence calculation functions.

    Examples:
        >>> pytest tests/test_criterions.py

"""

@pytest.mark.parametrize("class_specific", [False, True])
def test_cls_criterions(class_specific):
    criterions = moval.solvers.clsCalibrate(class_specific=class_specific)
    assert isinstance(criterions, moval.solvers.Calibrate)

    # classification logit input
    numclass = 10
    inp = np.random.randn(100, numclass) # 100 samples, 10 classes
    gt = np.random.randint(0, numclass, (100, ))
    if class_specific:
        estim = np.random.rand(numclass, )
    else:
        estim = np.random.rand(1, )[0]

    err = criterions(inp, gt, estim)
    if class_specific:
        assert len(err) == numclass
    assert np.mean(err) < 1
    assert np.mean(err) > 0

@pytest.mark.parametrize("class_specific", [False, True])
def test_seg_criterions(class_specific):
    criterions = moval.solvers.segCalibrate(class_specific=class_specific)
    assert isinstance(criterions, moval.solvers.Calibrate)

    # segmentation logit input
    numclass = 3
    inp = []
    gt = []
    for _ in range(5):
        inp.append(np.random.randn(numclass, 50, 50, 50)) # 5 samples, 3 classes, of volume shape ``(100, 100, 100)``
        gt.append(np.random.randint(0, numclass, (50, 50, 50)))
    if class_specific:
        estim = np.random.rand(numclass, )
    else:
        estim = np.random.rand(1, )[0]

    err = criterions(inp, gt, estim)
    if class_specific:
        assert len(err) == numclass
    assert np.mean(err) <= 1 and np.mean(err) >= 0
