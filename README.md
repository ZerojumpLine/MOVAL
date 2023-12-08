# MOVAL [![PyPI version](https://badge.fury.io/py/moval.svg)](https://badge.fury.io/py/moval)

**MOVAL** is a Python package designed for assessing model performance in the absence of ground truth labels. It computes and calibrated confidence scores to accurately reflect the likelihood of predictions, leveraging these calibrated confidence scores to estimate the model's overall performance. Notably, MOVAL operates without the need for ground truth labels in the target domains and supports the evaluation of model performance in classification, 2D segmentation, and 3D segmentation.

**MOVAL** highlights a key feature—class-wise calibration, recognized as essential for addressing long-tailed distributions commonly found in real-world datasets. This proves especially significant in segmentation tasks where background samples often outnumber foregrounds. The inclusion of class-specific variants becomes crucial for accurately estimating segmentation performance. Additionally, MOVAL offers support for various types of confidence scores, enhancing its versatility.

What it offers:
<br/> <div align=center><img src="https://github.com/ZerojumpLine/MOVAL/blob/main/docs/software_features.png?raw=True" width="750px"/></div>

## User Document

The latest documentation can be found [here](https://moval.readthedocs.io/en/latest/index.html).

## Reference

```
@inproceedings{li2022estimating,
  title={Estimating model performance under domain shifts with class-specific confidence scores},
  author={Li, Zeju and Kamnitsas, Konstantinos and Islam, Mobarakol and Chen, Chen and Glocker, Ben},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={693--703},
  year={2022},
  organization={Springer}
}
```