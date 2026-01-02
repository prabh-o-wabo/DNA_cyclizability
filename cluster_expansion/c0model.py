'''
Functions for computing W, J, and G as well as performing invariances
'''

from .helpers import *

import numpy as np
from numba import njit, prange

def trainW(onehotArray, c0Array, avgFreq = None):
    '''
    Function to solve for W (order 1 tensor) in the cluster expansion model given a set of DNA sequences.
    The W vector represents single-site contributions to DNA cyclizability

    Parameters
    ----------
    onehotArray: numpy array
        A 2D Numpy array containing onehot encoded DNA sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the onehotArray

    avgFreq : float = 0.25
        Expected frequency that a base of type 'alpha' exists at position 'i'
        Default is 0.25, assuming equal probability among all bases

    Returns
    -------
    Wvector : numpy array
        A 1D numpy array containing solved first-order coefficients for the set of sequences
    '''

    # ensure arrays are the same size
    assert len(onehotArray) == len(c0Array), "Arrays must have the same length"

    if avgFreq is None:
        avgFreq = np.full(200, 0.25, dtype = np.float32)
    else:
        avgFreq = np.asarray(avgFreq, dtype = np.float32)

    # initialize key values
    N = len(onehotArray)
    Wvector = np.zeros([200], dtype = np.float32)
    avgc0 = np.mean(c0Array)

    # center to the mean
    onehotArrayCtr = onehotArray - avgFreq
    c0ArrayCtr = c0Array - avgc0

    # solve for the w vector
    # -> see research paper for formula
    for i in range(200):
        Wvector[i] = 4 * np.inner(onehotArrayCtr[:, i], c0ArrayCtr) / N

    return Wvector.astype(np.float32)

def trainJ(onehotArray, c0Array, avgFreq = None):
    '''
    Function to solve for J (order 2 tensor) in the cluster expansion model given a set of DNA sequences.
    The J matrix represents second-order contributions to DNA cyclizability

    Parameters
    ----------
    onehotArray: numpy array
        A 2D Numpy array containing onehot encoded DNA sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the onehotArray

    avgFreq : float = 0.25
        Expected frequency that a base of type 'alpha' exists at position 'i'
        Default is 0.25, assuming equal probability among all bases

    Returns
    -------
    Jmatrix : numpy array
        A 2D numpy array containing solved second-order coefficients for the set of sequences
    '''

    # ensure arrays are the same size
    assert len(onehotArray) == len(c0Array), "Arrays must have the same length"

    if avgFreq is None:
        avgFreq = np.full(200, 0.25, dtype = np.float32)
    else:
        avgFreq = np.asarray(avgFreq, dtype = np.float32)

    # initialize key values
    N = len(onehotArray)
    Jmatrix = np.zeros([200, 200], dtype = np.float32)
    avgc0 = np.mean(c0Array)

    # center to the mean
    onehotArrayCtr = onehotArray - avgFreq
    c0ArrayCtr = c0Array - avgc0

    # solve for the J matrix
    # -> see research paper for formula
    for rowIdx in range(199):
        i, _ = SeparateBase(Idx2Base(rowIdx))

        # solve for upper triangle, then enforce symmetry
        for colIdx in range(rowIdx + 1, 200):
            j, _ = SeparateBase(Idx2Base(colIdx))

            # skip 'diagonal' entries --> i or j = 1 when row_ind or col_ind = {0,1,2,3}
            # terms captured in W
            if i == j:
                continue

            Jmatrix[ rowIdx, colIdx ] = 16 * InnerProduct(c0ArrayCtr, onehotArrayCtr[:, rowIdx], onehotArrayCtr[:, colIdx]) / N

            # impose symmetry
            Jmatrix[ colIdx, rowIdx ] = Jmatrix[ rowIdx, colIdx ]

    return Jmatrix.astype(np.float32)

def TranslationInvariance(Jmatrix):
    '''
    Function to apply translational invariance on the J matrix

    This helps reduce noise when smaller datasets are used to train

    Parameters
    ----------
    Jmatrix : numpy array
        A 2D Numpy array containing the second-order interaction parameters

    Returns
    -------
    Jmatrix_TI : numpy array
        A Translationally Invarianced J matrix
    '''

    assert Jmatrix.shape == (200, 200), "Jmatrix must be of shape 200 x 200"

    # sequence length
    N = 50

    # initialize the translationally invarianced J
    Jmatrix_TI = np.zeros_like(Jmatrix, dtype=np.float32)

    # solve for the J matrix
    # -> See research paper for formula
    for rowIdx in range(199):
        i, alpha = SeparateBase(Idx2Base(rowIdx))

        # solve for upper triangle, then enforce symmetry
        for colIdx in range(rowIdx + 1, 200):
            j, beta = SeparateBase(Idx2Base(colIdx))

            # separation between positions
            sep = j - i

            # if the separation is zero, skip
            if sep < 1:
                continue

            # use formula to apply translational invariance

            sum = 0
            for k in range(1, (N - sep) + 1):
                rowIdxNew = Base2Idx(f'{k}{alpha}')
                colIdxNew = Base2Idx(f'{k+sep}{beta}')

                sum += Jmatrix[ rowIdxNew, colIdxNew]

            Jmatrix_TI[ rowIdx, colIdx ] = sum / (N - sep)

            # enforce symmetry
            Jmatrix_TI[ colIdx, rowIdx ] = Jmatrix_TI[ rowIdx, colIdx]

    return Jmatrix_TI.astype(np.float32)

def ReverseComplementInvariance(Jmatrix_TI):
    '''
    Function to impose reverse complement invariance (J matrix must be translationally invarianced first)

    This helps reduce noise when smaller datasets are used to train

    Paramters
    ---------
    Jmatrix_TI : numpy array
        A 2D array representing a translationally invarianced J matrix

    Returns
    -------
    InvariancedJmatrix : numpy array
        A reverse complement and translationally invarianced J matrix
    '''

    assert Jmatrix_TI.shape == (200, 200), "Jmatrix_TI must be of shape 200 x 200"

    # initialize the fully invarianced J matrix
    InvariancedJmatrix = np.zeros_like(Jmatrix_TI, dtype=np.float32)

    # solve for the J matrix
    # -> See research paper for formula
    for rowIdx in range(200):
        for colIdx in range(200):

            # retrieve the indices for the reverse complement J matrix
            rowIdxNew, colIdxNew = ReverseComplementIndices(rowIdx, colIdx)

            # perform calculation for the reverse complement invariance
            InvariancedJmatrix[ rowIdx, colIdx ] = 0.5 * (Jmatrix_TI[rowIdx, colIdx] + Jmatrix_TI[rowIdxNew, colIdxNew])

    return InvariancedJmatrix.astype(np.float32)

def trainG(onehotArray, c0Array, avgFreq = None):
    '''
    Function to solve for G (order 3 tensor) in the cluster expansion model given a set of DNA sequences.
    The G tensor represents third-order contributions to DNA cyclizability

    Parameters
    ----------
    onehotArray: numpy array
        A 2D Numpy array containing onehot encoded DNA sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the onehotArray

    avgFreq : float = 0.25
        Expected frequency that a base of type 'alpha' exists at position 'i'
        Default is 0.25, assuming equal probability among all bases

    Returns
    -------
    Wvector : numpy array
        A 1D numpy array containing solved first-order coefficients for the set of sequences
    '''

    # ensure arrays are the same size
    assert len(onehotArray) == len(c0Array), "Arrays must have the same length"

    if avgFreq is None:
        avgFreq = np.full(200, 0.25, dtype = np.float32)
    else:
        avgFreq = np.asarray(avgFreq, dtype = np.float32)

    # initialize key values
    N = len(onehotArray)
    Gtensor = np.zeros([200, 200, 200], dtype = np.float32)
    avgc0 = np.mean(c0Array)

    # center to the mean
    onehotArrayCtr = onehotArray - avgFreq
    c0ArrayCtr = c0Array - avgc0

    # solve for the G tensor
    # -> see research paper for formula
    for rowIdx in range(198):
        i, _ = SeparateBase(Idx2Base(rowIdx))

        # solve for upper triangular 'prism', then enforce symmetry
        for colIdx in range(rowIdx + 1, 199):
            j, _ = SeparateBase(Idx2Base(colIdx))

            for depIdx in range(colIdx + 1, 200):
                k, _ = SeparateBase(Idx2Base(depIdx))

                # skip when the position is equal (terms captured in W and J)
                if i == j or i == k or j == k:
                    continue

                Gtensor[rowIdx, colIdx, depIdx] = 64 * InnerProduct(onehotArrayCtr[:, rowIdx], onehotArrayCtr[:, colIdx], onehotArrayCtr[:, depIdx], c0ArrayCtr) / N

                # impose symmetry across all permutations
                Gtensor[colIdx, rowIdx, depIdx] = Gtensor[rowIdx, colIdx, depIdx]
                Gtensor[rowIdx, depIdx, colIdx] = Gtensor[rowIdx, colIdx, depIdx]
                Gtensor[depIdx, rowIdx, colIdx] = Gtensor[rowIdx, colIdx, depIdx]
                Gtensor[colIdx, depIdx, rowIdx] = Gtensor[rowIdx, colIdx, depIdx]
                Gtensor[depIdx, colIdx, rowIdx] = Gtensor[rowIdx, colIdx, depIdx]
    
    return Gtensor.astype(np.float32)

@njit(fastmath=True)
def InnerProduct4_loop(a, b, c, y):
    '''
    Numba friendly inner product of 4 vectors
    '''
    s = 0.0
    for n in range(a.shape[0]):
        s += a[n] * b[n] * c[n] * y[n]
    return s

@njit(parallel=True, fastmath=True)
def trainG_numba(onehotArray, c0Array, avgFreq = None):
    '''
    Function to solve for G (order 3 tensor) in the cluster expansion model
    given a set of DNA sequences.

    The G tensor represents third-order contributions to DNA cyclizability
    '''

    # ensure arrays are the same size
    assert(len(onehotArray) == len(c0Array)), "Arrays must have the same length"

    if avgFreq is None:
        avgFreq = np.full(200, 0.25, dtype = np.float32)
    else:
        avgFreq = np.asarray(avgFreq, dtype = np.float32)

    # initialize key values
    N = len(onehotArray)
    Gtensor = np.zeros((200, 200, 200), dtype=np.float32)

    # compute mean(c0Array)
    avgC0 = np.mean(c0Array)

    # center to the mean
    onehotArrayCtr = onehotArray - avgFreq
    c0ArrayCtr = c0Array - avgC0

    # idx_to_pos: {0, ... , 200} -> {0, ... , 49}
    idx_to_pos = np.arange(200, dtype=np.int32) // 4

    # solve for the G tensor
    # -> see research paper for formula
    scale = np.float32(64.0 / N)

    # solve for upper triangular 'prism', then enforce symmetry
    for rowIdx in prange(198):
        i = idx_to_pos[rowIdx]

        for colIdx in range(rowIdx + 1, 199):
            j = idx_to_pos[colIdx]

            for depIdx in range(colIdx + 1, 200):
                k = idx_to_pos[depIdx]

                # skip when the position is equal (terms captured in W and J)
                if i == j or i == k or j == k:
                    continue

                val = scale * InnerProduct4_loop(
                    onehotArrayCtr[:, rowIdx],
                    onehotArrayCtr[:, colIdx],
                    onehotArrayCtr[:, depIdx],
                    c0ArrayCtr,
                )

                Gtensor[rowIdx, colIdx, depIdx] = val

                # impose symmetry across all permutations
                Gtensor[colIdx, rowIdx, depIdx] = Gtensor[rowIdx, colIdx, depIdx]
                Gtensor[rowIdx, depIdx, colIdx] = Gtensor[rowIdx, colIdx, depIdx]
                Gtensor[depIdx, rowIdx, colIdx] = Gtensor[rowIdx, colIdx, depIdx]
                Gtensor[colIdx, depIdx, rowIdx] = Gtensor[rowIdx, colIdx, depIdx]
                Gtensor[depIdx, colIdx, rowIdx] = Gtensor[rowIdx, colIdx, depIdx]

    return Gtensor

def TranslationInvariance_G(Gtensor):
    '''
    Function to apply translational invariance on the Gtensor

    This helps reduce noise when smaller datasets are used to train

    Parameters
    ----------
    Gtensor : numpy array
        A 3D Numpy array containing the third-order interaction parameters

    Returns
    -------
    Gtensor_TI : numpy array
        A Translationally Invarianced G tensor
    '''

    assert Gtensor.shape == (200, 200, 200), "Gtensor must be of shape 200 x 200 x 200"

    # sequence length
    N = 50

    # initialize the translationally invarianced J
    Gtensor_TI = np.zeros_like(Gtensor, dtype=np.float32)

    # solve for the G tensor
    # -> generalization of the J matrix function
    for rowIdx in range(198):
        i, alpha = SeparateBase(Idx2Base(rowIdx))

        for colIdx in range(rowIdx+1, 199):
            j, beta = SeparateBase(Idx2Base(colIdx))

            for depIdx in range(colIdx+1, 200):
                k, gamma = SeparateBase(Idx2Base(depIdx))

                # separation between positions
                sep_ji = j - i
                sep_ki = k - i

                # skip if separations are 0
                if sep_ji == 0 or sep_ki == 0:
                    continue
                
                # extract the largest separation
                maxSep = max(sep_ji, sep_ki)

                # use formula to apply translational invariance
                sum = 0
                for l in range(1, (N - maxSep) + 1):
                    rowIdxNew = Base2Idx(f'{l}{alpha}')
                    colIdxNew = Base2Idx(f'{l + sep_ji}{beta}')
                    depIdxNew = Base2Idx(f'{l + sep_ki}{gamma}')

                    sum += Gtensor[rowIdxNew, colIdxNew, depIdxNew]

                Gtensor_TI[ rowIdx, colIdx, depIdx] = sum / (N - maxSep)

                # Impose symmetry across all permutations of indices
                Gtensor_TI[colIdx, rowIdx, depIdx] = Gtensor_TI[rowIdx, colIdx, depIdx]
                Gtensor_TI[rowIdx, depIdx, colIdx] = Gtensor_TI[rowIdx, colIdx, depIdx]
                Gtensor_TI[depIdx, rowIdx, colIdx] = Gtensor_TI[rowIdx, colIdx, depIdx]
                Gtensor_TI[colIdx, depIdx, rowIdx] = Gtensor_TI[rowIdx, colIdx, depIdx]
                Gtensor_TI[depIdx, colIdx, rowIdx] = Gtensor_TI[rowIdx, colIdx, depIdx]

    return Gtensor_TI.astype(np.float32)

def ReverseComplementInvariance_G(Gtensor_TI):
    '''
    Function to impose reverse complement invariance (G tensor must be translationally invarianced first)

    This helps reduce noise when smaller datasets are used to train

    Paramters
    ---------
    Gtensor_TI : numpy array
        A 2D array representing a translationally invarianced J matrix

    Returns
    -------
    InvariancedGtensor : numpy array
        A reverse complement and translationally invarianced J matrix
    '''

    assert Gtensor_TI.shape == (200, 200, 200), "Gtensor_TI must be of shape 200 x 200 x 200"

    # initialize the fully invarianced G tensor
    InvariancedGtensor = np.zeros_like(Gtensor_TI, dtype=np.float32)

    # solve for the G tensor
    # -> generalization of the J matrix function
    for rowIdx in range(200):
        for colIdx in range(200):
            for depIdx in range(200):
                
                # retrieve the indicies for the reverse complement G tensor
                rowIdxNew, colIdxNew, depIdxNew = ReverseComplementIndices_G(rowIdx, colIdx, depIdx)

                # perform calculation for the reverse complement invariance
                InvariancedGtensor[rowIdx, colIdx, depIdx] = (1/2) * (Gtensor_TI[rowIdx, colIdx, depIdx] + Gtensor_TI[rowIdxNew, colIdxNew, depIdxNew])

    return InvariancedGtensor.astype(np.float32)


def Consolidate(W, J, G = None):
    '''
    Function to consolidate lower-order terms within higher-order tensors

    (ie. W in the diagonal of J or W and J in the skipped indices of G)

    Parameters
    ----------
    W : numpy array
        A 1D Numpy array containing the first-order interactions

    J : numpy array
        A 2D Numpy array containing the second-order interactions

    G : numpy array
        A 3D Numpy array containing the third-order interactions

    Returns
    -------
    consolidatedTensor : numpy array
        A 2D or 3D Numpy array where the coefficients for W are contained within J and J may be contained within G
    '''

    # ensure that the correct sizes are input
    assert (W.shape == (200,)) and (J.shape == (200,200)), 'W and J must be of size (200) and (200, 200), respectively'
    if G is not None:
        assert G.shape == (200, 200, 200), 'G must be of size (200, 200, 200)'

    # if G is not input, consolidate W witin J
    if G is None:
        
        # consolidate W along the diagonal
        for i in range(200):
            J[i, i] = 2 * W[i]
        return J
    
    # if G is input, consolidate W and J within G
    elif G is not None:

        Gconsolidated = np.zeros_like(G, dtype=np.float32)

        # consolidate J symmetrically within G

        for i in range(200):
            for j in range(200):
                for k in range(200):
                    
                    # include W along the diagonal
                    if i == j == k:
                        Gconsolidated[i,j,k] = 6 * W[i]

                    # always include an off-diagonal term for J
                    elif i == j:
                        Gconsolidated[i,j,k] = 2 * J[i,k]
                    elif k == i:
                        Gconsolidated[i,j,k] = 2 * J[k,j]
                    elif j == k:
                        Gconsolidated[i,j,k] = 2 * J[j,i]

                    else:
                        Gconsolidated[i,j,k] = G[i,j,k]

        return Gconsolidated.astype(np.float32)
    
    else:
        print('Something strange has happened')
        pass

def predictC0(onehotSeq, avgC0 = 0, avgFreq = None, W = np.zeros(200), J = np.zeros((200, 200)), G = np.zeros((200, 200, 200))):
    '''
    Function to predict a sequences cyclizability using the cluster expansion formula

    Paramters
    ---------
    onehotSeq : numpy array
        A 1D array containing a onehot encoded sequence
    
    avgc0 : float = 0
        The average cyclizability value associated with the sequences

    avgFreq : float = 0.25
        Expected frequency that a base of type 'alpha' exists at position 'i'
        Default is 0.25, assuming equal probability among all bases

    W : numpy array
        A 1D Numpy array containing the first-order interactions

    J : numpy array
        A 2D Numpy array containing the second-order interactions

    G : numpy array
        A 3D Numpy array containing the third-order interactions

    Returns
    -------
    c0 : float
        The predicted cyclizability associated with the sequence
    '''
    assert(len(onehotSeq) == 200), 'malformed sequence - onehot encoding must have length 200'

    if avgFreq is None:
        avgFreq = np.full(200, 0.25, dtype = np.float32)
    else:
        avgFreq = np.asarray(avgFreq, dtype = np.float32)

    # center data
    centered = (onehotSeq - avgFreq).astype(np.float32)

    # initialize prediction with average
    c0 = np.array(avgC0, dtype=np.float32)

    # Add contributions of W, J, and G
    c0 += (W @ centered)    \
        + ((1/2) * (centered @ J @ centered))   \
        + ((1/6) * (((np.tensordot(G, centered, axes = (2,0)) @ centered) @ centered)))

    return c0

def predictArrayC0(onehotArray, avgC0=0, avgFreq=None, W=np.zeros(200), J=np.zeros((200, 200)), G=np.zeros((200, 200, 200)), batchSize = 10_000):
    '''
    Function to predict sequence cyclizability for an array of sequences using the cluster expansion formula

    Parameters
    ---------
    onehotArray : numpy array
        A 2D array containing onehot encoded sequences
    
    avgc0 : float = 0
        The average cyclizability value associated with the sequences

    avgFreq : float = 0.25
        Expected frequency that a base of type 'alpha' exists at position 'i'
        Default is 0.25, assuming equal probability among all bases

    W : numpy array
        A 1D Numpy array containing the first-order interactions

    J : numpy array
        A 2D Numpy array containing the second-order interactions

    G : numpy array
        A 3D Numpy array containing the third-order interactions

    Returns
    -------
    c0Array : numpy array
        The predicted cyclizabilities associated to each sequence entered
    '''    
    assert onehotArray.shape[1] == 200

    if avgFreq is None:
        avgFreq = np.full(200, 0.25, dtype = np.float32)
    else:
        avgFreq = np.asarray(avgFreq, dtype = np.float32)

    X_full = (onehotArray - avgFreq).astype(np.float32)
    N = len(X_full)
    c0Array = np.empty(N, dtype=np.float32)

    use_J = np.any(J)
    use_G = np.any(G)

    for s in range(0, N, batchSize):
        e = min(s + batchSize, N)
        X = X_full[s:e]                     # (b, 200)
        # first-order
        t1 = X @ W                          # (b,)

        # second-order
        t2 = np.zeros(len(X), dtype=np.float32)
        if use_J:
            JX = X @ J                      # (b,200)
            t2 = np.sum(JX * X, axis=1)     # (b,)

        # third-order using tensordot (kept as requested)
        t3 = np.zeros(len(X), dtype=np.float32)
        if use_G:
            # GX: (200, 200, b)
            GX = np.tensordot(G, X, axes=(2, 1))
            # sum over middle index -> (200, b)
            GX = np.sum(GX * X.T[None, :, :], axis=1)
            # contract with X.T -> (b,)
            t3 = np.sum(GX * X.T, axis=0)

        c0_batch = np.full(len(X), avgC0, dtype=np.float32)
        c0_batch += t1
        c0_batch += 0.5 * t2
        c0_batch += (1.0 / 6.0) * t3

        c0Array[s:e] = c0_batch

    return c0Array.astype(np.float32)