Usage
===================================


Using MOVAL
----------------

MOVAL provides support for a range of confidence score and model calibration techniques. 
It is easily applicable to both classification and segmentation tasks, particularly in safety-critical applications such as medical image analysis.


Quick Starts
----------------

To optimize MOVAL parameter and estimate the performance on a classification task:

>>> import moval
>>> import numpy as np
>>> logits = np.random.randn(1000, 10)
>>> gt = np.random.randint(0, 10, (1000))
>>> moval_model = moval.MOVAL()
>>> moval_model.fit(logits, gt)
>>> estim_acc = moval_model.estimate(logits)

More pratical cases are shown in :doc:`tutorial`.