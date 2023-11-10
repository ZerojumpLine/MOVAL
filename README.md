# MOVAL

## Introduction

A python package to evaluate model performance without the ground truth label.


## Installation

For Conda users, you can create a new Conda environment using

```
conda create -n moval python=3.8
```

and then intall all the dependencies with

```
pip install -r requirements.txt
```

## Usage

example:

logits = np.random.uniform(0, 1, (1000, 10))
gt = np.random.randint(0, 10, (1000))
moval_model = moval.MOVAL(
    mode = "classification",
    confidence_scores = "raw",
    estim_algorithm = "ac",
    class_specific = False
    )
moval_model.fit(logits, gt)
estim_acc = moval_model.estimate(logits)