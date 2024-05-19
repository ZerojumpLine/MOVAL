

"""MOVAL is tool to estimate model performance without manual labels. 
It contains several confidences score calculation and calibration algorithms, implemented with python and sklearn.
It can estimate accuracy for classification tasks and dice scores for segmentation tasks.
"""

try:
    from moval.integrations.sklearn.moval import MOVAL
except ImportError as e:
    # silently fail for now
    pass

__version__ = "0.3.10"
__all__ = ["MOVAL"]

def __getattr__(key):
    """Lazy import of moval submodules and -packages.

    Once :py:mod:`moval` is imported, it is possible to lazy import

    """
    if key == "MOVAL":
        from moval.integrations.sklearn.moval import MOVAL

        return MOVAL