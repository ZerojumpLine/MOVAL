import pytest
import numpy as np

import moval
import moval.models

"""Test the confidence calculation functions.

    Examples:
        >>> pytest tests/test_confidences.py

"""

@pytest.mark.parametrize("confidence_scores", moval.models.get_conf_options())
def test_conf(confidence_scores):
    conf = moval.models.init(confidence_scores)
    assert isinstance(conf, moval.models.Confidence)

    # classification logit input
    inp = np.random.randn(100, 10) # 100 samples, 10 classes
    score = conf(inp)
    assert isinstance(score, np.ndarray)
    assert score.shape == (100, )

    if not conf.normalization:
        assert np.max(score) <= 1
        assert np.min(score) >= 0

    inp_large_var = 10 * inp # logit with higher confidence
    score_large_var = conf(inp_large_var)
    assert np.min(score_large_var - score) > 0
    # should be more confidence!

    # segmentation logit input
    inp = []
    for _ in range(10):
        inp.append(np.random.randn(3, 100, 100, 100)) # 10 samples, 3 classes, of volume shape ``(100, 100, 100)``
    score = conf(inp)
    assert isinstance(score, list)
    assert score[0].shape == (100, 100, 100)

    if not conf.normalization:
        assert np.max(score) <= 1
        assert np.min(score) >= 0