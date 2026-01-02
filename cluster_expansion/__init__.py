# This file marks the cluster_expansion directory as a Python package.
# You can import modules from this package using:
# from cluster_expansion import <module_name>

# outside packages
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from numba import njit, prange
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split

# personal library
from .batchTraining import trainW10k, trainJ10k, trainG10k
from .c0model import *
from .CNN import *
from .encoding import Data2Onehot, Onehot2Data, SeqArr2RevCompArr
from .helpers import Project2Sequence, envelope, trim, split9010
from .testing import c0Test, crossValidationSplits, plot_hist2d, pearson

__version__ = '1.0'