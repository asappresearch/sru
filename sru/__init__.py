"""
sru - Simple Recurrent Unit

sru provides a PyTorch implementation of the simple recurrent neural network cell described
in "Simple Recurrent Units for Highly Parallelizable Recurrence."
"""
from .version import __version__
try:
    from .sru_functional import *
except:
    from sru_functional import *
