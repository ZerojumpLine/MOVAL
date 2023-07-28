

"""
MOVAL is tool to estimate model performance without manual labels.
"""

def __getattr__(key):
    """Lazy import of moval submodules and -packages.

    Once :py:mod:`moval` is imported, it is possible to lazy import

    """
    if key == "MOVAL":
        from moval.moval import MOVAL

        return MOVAL