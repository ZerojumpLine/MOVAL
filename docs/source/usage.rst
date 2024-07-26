Usage
===================================


Using MOVAL
----------------

MOVAL provides support for a range of confidence score and model calibration techniques. 
It is easily applicable to both classification and segmentation tasks, particularly in safety-critical applications such as medical image analysis.


Quick Starts with Classification Tasks
----------------

To optimize MOVAL parameter and estimate the accuracy on a classification task:

>>> import moval
>>> import numpy as np
>>> logits = np.random.randn(1000, 10)
>>> gt = np.random.randint(0, 10, (1000))
>>> l_test = np.random.randn(1000, 10)
>>> moval_model = moval.MOVAL()
>>> moval_model.fit(logits, gt)
>>> estim = moval_model.estimate(l_test)

This will return the default estimating method, which is average confidence of max class probability.

The MOVAL moval are alternatively defined with:

>>> model = moval.MOVAL(
    mode = "classification",
    metric = "accuracy",
    confidence_scores = "entropy-conf",
    estim_algorithm = "ts-model",
    class_specific = False
)

or the essemble one:

>>> model = moval.MOVAL(
    mode = "classification",
    estim_algorithm = \
    "moval-ensemble-cls-accuracy"
)

Quick Starts with Segmentation Tasks
----------------
Estimating the performance (F1-Score) of 3D segemntation models are also straightforward:

>>> import moval
>>> import numpy as np
>>> C = 2 ; H = 50 ; W = 50 ; D = 80
>>> logits = [] ; gts = []
>>> for _ in range(5):
        logit = np.random.randn(C, H, W, D)
        gt = np.random.randint(0, C, (H, W, D))
        logits.append(logit)
        gts.append(gt)
>>> l_test = [np.random.randn(C, H, W, D)]
>>> model = moval.MOVAL(
    mode = "segmentation",
    metric = "f1score",
    confidence_scores = "doctor-conf",
    estim_algorithm = "atc-model",
    class_specific = True
)
>>> moval_model.fit(logits, gt)
>>> estim = moval_model.estimate(l_test)

2D segmentation can be achieved in a very similar way, just with a different input tensor shape. More pratical cases are shown in :doc:`tutorial`.