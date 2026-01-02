'''
Functions to encode DNA sequences into usable data
'''

import numpy as np
import pandas as pd

def Seq2Onehot(sequence: str, customMap = None):
    '''
    Onehot-encode a DNA sequence to a flattened array

    Each base is mapped to a 4-element onehot vector:
        A -> [1,0,0,0]
        T -> [0,1,0,0]
        C -> [0,0,1,0]
        G -> [0,0,0,1]

    the result is then concatenated and flattened into a 1D array

    Parameters
    ----------
    sequence : str
        A string of DNA bases (ATCG) - expected length is 50

    Return
    ------
    onehotSeq : numpy array
        1D numpy array of expected length 200 containing the onehot encoded DNA sequence
    '''

    if len(sequence) != 50:
        raise ValueError(f'Sequence must be exactly 50 bases, received {len(sequence)} bases.')

    for idx, base in enumerate(sequence):
        if base not in 'ATCG':
            raise ValueError(f'Unidentified base found at position {idx + 1}: {base}')

    # Mapping from DNA -> Onehot
    OnehotMap = {
        'A' : [1,0,0,0],
        'T' : [0,1,0,0],
        'C' : [0,0,1,0],
        'G' : [0,0,0,1]
    }

    if customMap is not None:
        OnehotMap = customMap

    # convert all bases to onehot vectors
    unflattened_onehot = [OnehotMap[base] for base in sequence]

    # flatten
    onehotSeq = np.array(unflattened_onehot, dtype = np.int8).flatten(order = 'C')

    return onehotSeq

def Data2Onehot(sequenceArray, customMap = None, verbose = True):
    '''
    Onehot encode an array of sequence - each of length 50

    Parameters
    ----------
    sequenceArray : pandas dataframe or numpy array
        A 1D array/dataframe containing string representations of DNA sequences

    Returns
    -------
    onehotArray : numpy array
        A 2D numpy array where each row is a onehot encoded sequence
    '''
    try:
        # iterate through the dataframe, adding onehot sequences to the list
        onehotList = []
        for sequence in sequenceArray:
            onehotSeq = Seq2Onehot(sequence, customMap)
            onehotList.append(onehotSeq)

        # join the sequence of arrays together
        onehotArray = np.stack(onehotList, axis=0).astype(np.int8)

        if verbose:
            print('Encoded all sequences to onehot')
        return onehotArray

    except Exception as e:

        if not isinstance(e, ValueError):
            print(e)

        # determine which sequence the error occurs in
        for idx, sequence in enumerate(sequenceArray):
            try:
                Seq2Onehot(sequence)
            except ValueError as ve:
                print(f'Error with sequence {idx + 1}: {sequence}\n{ve}')
        
        return np.zeros((len(sequenceArray), 200), dtype=np.int8)

def Onehot2Data(onehotArray, customMap = None):
    '''
    Convert a set of onehot encodes sequences to an array of sequences in string form
        - This is a reversed version of Data2Onehot

    Parameters
    ----------
    onehotArray : numpy or pandas array
        A 2D numpy array containing flattened onehot encoded sequences

    Returns
    -------
    sequenceArray : numpy array
        A 1D array containing string representations of DNA sequences
    '''

    try:
        arr = np.asarray(onehotArray)

        # single sequence
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        
        # malformed input
        if arr.ndim != 2 or arr.shape[1] != 200:
            raise ValueError("onehotArray must have shape (N, 200) or (200,)")
        
        # capture index of "1" in the onehot encoding
        idx = arr.reshape(arr.shape[0], 50, 4).argmax(axis=2)

        base_map = np.array(list('ATCG'))
        # custom mapping
        if customMap is not None:
            base_map = np.array(list(customMap))
        
        # apply mapping and convert each row to a sequence string
        chars = base_map[idx]
        seqs = [''.join(row) for row in chars]
        sequenceArray = np.array(seqs, dtype='<U50')

        return sequenceArray

    except Exception as e:

        print(e)

        return np.zeros((len(onehotArray)))

def SeqArr2RevCompArr(sequenceArray):
    '''
    Creates an array containing a set of sequences reverse complement to the input array

    Paramters
    ---------
    sequenceArray : pandas dataframe or numpy array
        A 1D array/dataframe containing string representations of DNA sequences

    Returns
    -------
    rcArray : pandas dataframe or numpy array
        A 1D array/dataframe containing string representations of DNA sequences reverse complement to the input
    '''

    complement = str.maketrans('ATCG', 'TAGC')

    rcArray = [seq.translate(complement)[::-1] for seq in sequenceArray]
    return np.array(rcArray)

if __name__ == '__main__':
    pass
    
