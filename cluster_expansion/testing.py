'''
Functions to test the Cluster Expansion model accuracy
'''

from .helpers import *
from .c0model import *
from .encoding import *

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm

def c0Test(onehotArray, c0Array, avgFreq = None, avgC0 = None, W = np.zeros((200)), J = np.zeros((200, 200)), G = np.zeros((200, 200, 200)), verbose = True):
    '''
    Predicts
    '''
    assert len(onehotArray) == len(c0Array), f"len(onehotArray) ({len(onehotArray)})≠ len(c0Array) ({len(c0Array)})"

    # predict C0
    if avgC0 == None:
        avgC0 = 0
    
    if avgFreq is None:
        avgFreq = np.full(200, 0.25, dtype = np.float32)
    else:
        avgFreq = np.asarray(avgFreq, dtype = np.float32)

    c0PredictionsArr = predictArrayC0(onehotArray, avgFreq = avgFreq, avgC0 = avgC0, W = W, J = J, G = G)

    # gather correlation coefficients
    r, se_z, prec, d_low, d_high = pearson(c0Array, c0PredictionsArr)

    # print out averages
    if verbose:
        print(f'Pearson correlation: {r:.{prec}f}, [{d_low:.{prec}f}, {d_high:.{prec}f}]')

    # return predictions
    return (c0PredictionsArr, r, (d_low, d_high), prec)

def crossValidationSplits(onehotArray, c0Array, iterations = 1, trainWvector = False, trainJmatrix = False, trainGtensor = False, invarianceJ = False, invarianceG = False):
    '''
    Function to perform iterative cross-validation tests

    Paramters
    ---------
    onehotArray : numpy array
        A 2D array containing onehot encoded sequences
    
    c0Array : numpy array
        A 1D numpy array containing cyclizability values associated tot he onehotArray

    iterations: int
        The number of times to test the data (higher for greater accuracy)

    trainW, trainJ, & trainG : bool
        A setting for whether or not to train W, J, and G on the sequences

    invarianceJ & invarianceG : bool
        A setting to choose whether or not to invariance the J matrix and/or G tensor

    Returns
    -------
    None -> Prints all relevant information
    '''

    assert len(onehotArray) == len(c0Array), 'the number of sequences and the number of cyclizability values must be equal'

    assert iterations >= 1, 'There must be at least 1 iteration'

    # get 10% of the size of the data
    iterationSize = int(np.ceil(len(onehotArray) * 0.1))

    # initialize containers to store correlation coefficients
    rVals = []
    rErrors = []

    results = {}

    for iter in range(iterations):

        (onehotTrainArray, onehotTestArray), (c0Train, c0Test) = split9010(onehotArray, c0Array, testing=0.1)

        # Training

        W = np.zeros((200))
        J = np.zeros((200, 200))
        G = np.zeros((200, 200, 200))

        if trainWvector:
            W = trainW(onehotTrainArray, c0Train)

        if trainJmatrix:
            J = trainJ(onehotTrainArray, c0Train)

            if invarianceJ:
                J = ReverseComplementInvariance(TranslationInvariance(J))

        if trainGtensor:
            G = trainG_numba(onehotTrainArray, c0Train)

            if invarianceG:
                G = ReverseComplementInvariance_G(TranslationInvariance_G(G))

        # Testing/Predicting

        avgc0 = np.mean(c0Train)
        c0Predict = predictArrayC0(onehotTestArray, avgc0, W=W, J=J, G=G)

        # determine pearson coefficients
        r, se_z, prec, d_low, d_high = pearson(c0Test.T, c0Predict.T)

        # append to containers
        rVals.append(r)
        rErrors.append((d_low, d_high))

        # add new sequences to the map
        seqs = Onehot2Data(onehotTestArray)
        for seq, meas, pred in zip(seqs, c0Test, c0Predict):
            if seq not in results:
                results[seq] = (meas, pred)

    # print out averages
    r = np.mean(rVals)
    rErr = (np.mean([err[0] for err in rErrors]), np.mean([err[1] for err in rErrors]))
    print(f'Pearson correlation: {r:.{prec}f}, [{rErr[0]:.{prec}f}, {rErr[1]:.{prec}f}]')

    # return predictions
    sequences = np.array(list(results.keys()))
    measuredC0s = np.array([v[0] for v in results.values()], dtype = np.float32)
    predictedC0s = np.array([v[1] for v in results.values()], dtype = np.float32)

    return (sequences, measuredC0s, predictedC0s), (r, rErr, prec)

def plot_hist2d(measuredC0s, predictedC0s, r, rErr, vmin, vmax, fig_dim = 5, n_bins = 50, range = (-1, 1), save = False, filename = None, precision = 3):
  tick_step = (vmax - vmin) // 5
  ticks_arr = np.arange(vmin, vmax + tick_step, tick_step)
  plt.figure(figsize=(fig_dim, fig_dim))

  plt.hist2d(measuredC0s, predictedC0s, bins = n_bins, cmap = 'hot_r', range=[list(range), list(range)], vmin = vmin, vmax = vmax)
  plt.colorbar(label = 'Count', shrink = 1.0)
  
  plt.xlim(range[0], range[1])
  plt.ylim(range[0], range[1])
  plt.xticks(np.linspace(range[0], range[1], 5), fontsize=12)
  plt.yticks(np.linspace(range[0], range[1], 5), fontsize=12)

  plt.xlabel('Measured C$_0$', fontsize = 14)
  plt.ylabel('Predicted C$_0$', fontsize = 14)
  plt.title('Correlation of Cluster Expansion Model against Measured C$_0$')
  plt.text(-0.95, 0.95, f'$r = {r:.{precision}f}, [{rErr[0]:.{precision}f}, {rErr[1]:.{precision}f}]$', fontsize = 14, verticalalignment='top', bbox=dict(boxstyle = 'round', facecolor = 'white', alpha = 0.7))
  plt.tight_layout()

  if (save and ((filename is not None) and (isinstance(filename, str)))):
    plt.savefig(f'{filename}.png')
  plt.show()

import numpy as np
from scipy.stats import pearsonr, norm

def pearson(measured, predicted, ci_int=0.95, out=False):
    """
    Return Pearson r and the CI offsets relative to r:
      - d_low  = ci.low  - r   (typically negative)
      - d_high = ci.high - r   (typically positive)

    Also returns Fisher-z SE and a suggested print precision.
    """
    res = pearsonr(measured, predicted)

    r = float(res.statistic)
    ci = res.confidence_interval(confidence_level=ci_int)

    d_low = float(ci.low - r)
    d_high = float(ci.high - r)

    # Fisher-z SE (in z-space), inferred from CI width
    z_low = np.arctanh(ci.low)
    z_high = np.arctanh(ci.high)
    zcrit = norm.ppf((1 + ci_int) / 2)
    se_z = float((z_high - z_low) / (2 * zcrit))

    # precision based on a typical magnitude of the offsets
    pm_scale = max(abs(d_low), abs(d_high))
    precision = max(0, int(np.ceil(-np.log10(pm_scale)))) if pm_scale > 0 else 0

    if out:
        print(f"r = {r:.{precision}f}, [{d_low:.{precision}f}, {d_high:.{precision}f}] (for {ci_int:.0%} CI)")

    return r, se_z, precision, d_low, d_high,




if __name__ == "__main__":
    pass