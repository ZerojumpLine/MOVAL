import pytest
import numpy as np

import moval

"""Test the save and load functions.

    Examples:
        >>> pytest tests/10_test_load.py

"""

def test_load_normal():
    moval_model = moval.MOVAL(
                        mode = "classification",
                        confidence_scores = "max_class_probability-conf",
                        estim_algorithm = "ts-model",
                        class_specific = False
                        )
    logits = np.random.randn(1000, 10)
    gt = np.random.randint(0, 10, (1000))
    moval_model.fit(logits, gt)
    moval_model.save('./foo.pkl')
    loaded_model = moval.MOVAL.load('./foo.pkl')
    estim_acc = loaded_model.estimate(logits)
    assert estim_acc <= 1 and estim_acc >= 0

def test_load_ensemble():
    # also test the ensemble func
    moval_model = moval.MOVAL(
                        mode = "classification",
                        confidence_scores = "max_class_probability-conf", # doesnt matter
                        estim_algorithm = "moval-ensemble",
                        class_specific = True # doesnt matter
                        )
    logits = np.random.randn(1000, 10)
    gt = np.random.randint(0, 10, (1000))
    moval_model.fit(logits, gt)
    moval_model.save('./foo.pkl')
    loaded_model = moval.MOVAL.load('./foo.pkl')
    estim_acc = loaded_model.estimate(logits)
    assert estim_acc <= 1 and estim_acc >= 0
