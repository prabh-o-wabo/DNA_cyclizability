'''
Functions to solve for the J matrix and the G tensor in batch sizes
'''

from .c0model import *

import numpy as np # pyright: ignore[reportMissingImports]

def trainW10k(onehotArray, c0Array, batchSize = 10000, updates = False, update_int = 5):
    '''
    Function to train the W tensor in batch sizes (default 10k)

    Parameters
    ----------
    onehotArray: numpy array
        A 2D Numpy array containing onehot encoded DNA sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the onehotArray
    
    batchSize : int
        The number of sequences to train the J matrix on per 'run'

    progressUpdate : bool = False
        A setting to print statements showing training progress

    Returns
    -------
    Wtensor : numpy array
        A 1D Numpy array capturing second-order interactions
    '''

    # initialize data
    N = len(onehotArray)
    numBatches = (N // batchSize)
    numRemainder = (N % batchSize)

    # ensure parameters are proper
    assert len(onehotArray) == len(c0Array), f'Input arrays must have the same length, {len(onehotArray)} ≠ {len(c0Array)}'

    # initialize container to store J matrices
    Wtensor = np.zeros((200), dtype=np.float32)
    processed_data = 0

    # iterate over the number of batches
    for (batch) in range(numBatches):

        # indices to extract a batch of the data
        startIdx = batch * batchSize
        endIdx = (batch + 1) * batchSize

        # extract batches
        onehotData = onehotArray[startIdx : endIdx]
        c0Data = c0Array[startIdx : endIdx]

        # sequences in batch
        n = len(onehotData)

        W_batch = trainW(onehotData, c0Data).astype(np.float32)
        Wtensor += W_batch * n
        processed_data += n

        # optional progress updates every 5 batches
        if updates and (batch % update_int == 0):
            print(f'Trained {batch + 1} of {numBatches} batches')

    # if there are extra sequences
    if (numRemainder):
        startIdx = numBatches * batchSize
        onehotData = onehotArray[startIdx:]
        c0Data = c0Array[startIdx:]

        n = len(onehotData)

        W_batch = trainW(onehotData, c0Data).astype(np.float32)
        Wtensor += W_batch * n
        processed_data += n

        if updates:
            print(f'Trained remainder batch ({n} sequences)')

    # divide by the total number of sequences "trained" on
    assert processed_data == N, f'All data not processed ( only {processed_data} of {N} processed)'
    Wtensor /= N

    return Wtensor.astype(np.float32)

def trainJ10k(onehotArray, c0Array, batchSize = 10000, updates = False, update_int = 5):
    '''
    Function to train the J matrix in batch sizes (default 10k)

    Parameters
    ----------
    onehotArray: numpy array
        A 2D Numpy array containing onehot encoded DNA sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the onehotArray
    
    batchSize : int
        The number of sequences to train the J matrix on per 'run'

    progressUpdate : bool = False
        A setting to print statements showing training progress

    Returns
    -------
    Jmatrix : numpy array
        A 2D Numpy array capturing second-order interactions
    '''

    # initialize data
    N = len(onehotArray)
    numBatches = (N // batchSize)
    numRemainder = (N % batchSize)

    # ensure parameters are proper
    assert len(onehotArray) == len(c0Array), f'Input arrays must have the same length, {len(onehotArray)} ≠ {len(c0Array)}'

    # initialize container to store J matrices
    Jmatrix = np.zeros((200, 200), dtype=np.float32)
    processed_data = 0

    # iterate over the number of batches
    for (batch) in range(numBatches):

        # indices to extract a batch of the data
        startIdx = batch * batchSize
        endIdx = (batch + 1) * batchSize

        # extract batches
        onehotData = onehotArray[startIdx : endIdx]
        c0Data = c0Array[startIdx : endIdx]

        # sequences in batch
        n = len(onehotData)

        J_batch = trainJ(onehotData, c0Data).astype(np.float32)
        Jmatrix += J_batch * n
        processed_data += n

        # optional progress updates every 5 batches
        if updates and (batch % update_int == 0):
            print(f'Trained {batch + 1} of {numBatches} batches')

    # if there are extra sequences
    if (numRemainder):
        startIdx = numBatches * batchSize
        onehotData = onehotArray[startIdx:]
        c0Data = c0Array[startIdx:]

        n = len(onehotData)

        J_batch = trainJ(onehotData, c0Data).astype(np.float32)
        Jmatrix += J_batch * n
        processed_data += n

        if updates:
            print(f'Trained remainder batch ({n} sequences)')

    # divide by the total number of sequences "trained" on
    assert processed_data == N, f'All data not processed ( only {processed_data} of {N} processed)'
    Jmatrix /= N

    return Jmatrix.astype(np.float32)

def trainG10k(onehotArray, c0Array, batchSize = 10000, updates = False, update_int = 5):
    '''
    Function to train the G tensor in batch sizes (default 10k)

    Parameters
    ----------
    onehotArray: numpy array
        A 2D Numpy array containing onehot encoded DNA sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the onehotArray
    
    batchSize : int
        The number of sequences to train the J matrix on per 'run'

    progressUpdate : bool = False
        A setting to print statements showing training progress

    Returns
    -------
    Gtensor : numpy array
        A 3D Numpy array capturing second-order interactions
    '''

    # initialize data
    N = len(onehotArray)
    numBatches = (N // batchSize)
    numRemainder = (N % batchSize)

    # ensure parameters are proper
    assert len(onehotArray) == len(c0Array), f'Input arrays must have the same length, {len(onehotArray)} ≠ {len(c0Array)}'

    # initialize container to store J matrices
    Gtensor = np.zeros((200, 200, 200), dtype=np.float32)
    processed_data = 0

    # iterate over the number of batches
    for (batch) in range(numBatches):

        # indices to extract a batch of the data
        startIdx = batch * batchSize
        endIdx = (batch + 1) * batchSize

        # extract batches
        onehotData = onehotArray[startIdx : endIdx]
        c0Data = c0Array[startIdx : endIdx]

        # sequences in batch
        n = len(onehotData)

        G_batch = trainG_numba(onehotData, c0Data).astype(np.float32)
        Gtensor += G_batch * n
        processed_data += n

        # optional progress updates every 5 batches
        if updates and (batch % update_int == 0):
            print(f'Trained {batch + 1} of {numBatches} batches')

    # if there are extra sequences
    if (numRemainder):
        startIdx = numBatches * batchSize
        onehotData = onehotArray[startIdx:]
        c0Data = c0Array[startIdx:]

        n = len(onehotData)

        G_batch = trainG_numba(onehotData, c0Data).astype(np.float32)
        Gtensor += G_batch * n
        processed_data += n

        if updates:
            print(f'Trained remainder batch ({n} sequences)')

    # divide by the total number of sequences "trained" on
    assert processed_data == N, f'All data not processed ( only {processed_data} of {N} processed)'
    Gtensor /= N

    return Gtensor.astype(np.float32)