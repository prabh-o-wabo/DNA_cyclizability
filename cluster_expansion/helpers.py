'''
Functions to assist with other functions in the cluster_expansion library
'''

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

def InnerProduct(*vectors):
    '''
    Function to take the inner product (elementwise product summed) of an arbitrary number of vectors

    Parameters
    ----------
    *vectors : numpy arrays
        An arbitrary number of 1D numpy arrays

    Returns
    -------
    dotProduct : int/float
        the inner product
    '''
    assert len(vectors) >= 2, 'Must be at least two vectors to calculate the inner product'
    # Elementwise multiply all vectors, then sum
    result = vectors[0]
    for v in vectors[1:]:
        result = result * v
    return np.sum(result)
    
def Idx2Base(idx):
    '''
    Converts a (0-199) index to its 1A, 1T, ... , 50C, 50G representation
    (ie. 6 -> '2C')

    Parameters
    ----------
    idx : int
        a (0-199) index representing i_alpha

    Returns
    -------
    i_alpha : str
        a 1A, 1T, ... , 50C, 50G position-base pairing
    '''

    assert 0 <= idx < 200, 'index must be in range 0 to 199'

    BaseMap = {
        0: 'A',
        1: 'T',
        2: 'C',
        3: 'G'
    }

    # solve for position
    i = (idx // 4) + 1

    # solve for the base
    alpha = BaseMap[idx % 4]

    # concatenate to get position-base pairing
    return f"{i}{alpha}"

def SeparateBase(pairing: str):
    '''
    Function to separate a base into its componenets 
    
    (ie. 2C -> (2, 'C'))

    Parameters
    ----------
    pairing : str
        a 1A, 1T, ... , 50C, 50G position-base pairing

    Returns
    -------
    i : int
        the position
    alpha : str
        the base
    '''

    assert len(pairing) == 2 or len(pairing) == 3, 'pairing can only be up to 2 position digits and 1 base character'

    # extract the position
    i = int(pairing[:-1])

    # extract the base
    alpha = pairing[-1]
    return i, alpha

def Base2Idx(pairing: str):
    '''
    Function to convert a (1A, 1T, ... , 50C, 50G) pairing to its (0-199) index representation

    (ie. '2C' -> 6)

    Parameters
    ----------
    pairing : str
        a 1A, 1T, ... , 50C, 50G position-base pairing

    Returns
    -------
    idx : int
        a (0-199) index representing i_alpha
    '''

    assert len(pairing) == 2 or len(pairing) == 3, 'pairing can only be up to 2 position digits and 1 base character'


    ReverseBaseMap = {
        'A' : 0,
        'T' : 1,
        'C' : 2,
        'G' : 3
    }

    # split into position and base components
    dnaPos, base = SeparateBase(pairing)
    
    # solve for indices of dna location of base base
    dnaIdx = (dnaPos - 1) * 4
    baseIdx = ReverseBaseMap[base]

    # add together to get index representation
    idx = dnaIdx + baseIdx
    return idx

def ReverseComplementIndices(idx1, idx2):
    '''
    Function to get the indices used to perform reverse complement invariance

    Parameters
    ----------
    idx1, idx2 : int
        indices for i_alpha and j_beta

    Returns
    -------
    idx1New, idx2New : int
        indices for reverse complement invariance
    '''

    # length of the sequence
    N = 50

    # map of what base is the reverse complement of what other base
    ReverseComplementMap = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C'
    }

    # extract position and bases
    i, alpha = SeparateBase(Idx2Base(idx1))
    j, beta = SeparateBase(Idx2Base(idx2))

    # get the new positions and bases
    iNew = str(N-j+1)
    jNew = str(N-i+1)
    betaPrime = ReverseComplementMap.get(beta)
    alphaPrime = ReverseComplementMap.get(alpha)

    # convert back to (0-199) index representation
    idx1New = Base2Idx(f'{iNew}{betaPrime}')
    idx2New = Base2Idx(f'{jNew}{alphaPrime}')

    return idx1New, idx2New

def ReverseComplementIndices_G(idx1, idx2, idx3):
    '''
    Function to get the indices used to perform reverse complement invariance on the G tensor

    Parameters
    ----------
    idx1, idx2, idx3 : int
        indices for i_alpha and j_beta and k_gamma

    Returns
    -------
    idx1New, idx2New, idx3New : int
        indices for reverse complement invariance
    '''
    
    # length of the sequence
    N = 50

    # map of what base is the reverse complement of what other base
    ReverseComplementMap = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C'
    }

    # extract position and bases
    i, alpha = SeparateBase(Idx2Base(idx1))
    j, beta = SeparateBase(Idx2Base(idx2))
    k, gamma = SeparateBase(Idx2Base(idx3))

    # get the new positions and bases
    iNew = str(N-k+1)
    jNew = str(N-j+1)
    kNew = str(N-i+1)
    gammaPrime = ReverseComplementMap.get(gamma)
    betaPrime = ReverseComplementMap.get(beta)
    alphaPrime = ReverseComplementMap.get(alpha)

    # convert back to (0-199) index representation
    idx1New = Base2Idx(f'{iNew}{gammaPrime}')
    idx2New = Base2Idx(f'{jNew}{betaPrime}')
    idx3New = Base2Idx(f'{kNew}{alphaPrime}')

    return idx1New, idx2New, idx3New

def split9010(onehotArray, c0Array, testing = 0.1):
    '''
    Helper function to split data into testing and training data

    Paramters
    ---------
    onehotArray: numpy array
        A 2D Numpy array containing onehot encoded DNA sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the onehotArray

    testing : int/float = 10
        The fraction of the data that will be partitioned for testing

    Returns
    -------
    onehotTrainArray, onehotTestArray : numpy arrays
        2D Numpy arrays containing onehot sequences for training and testing, respectively
    
    c0Train, c0Test : numpy arrays
        1D Numpy arrays containing predicted cyclizability values for training and testing, respectively
    '''

    onehotTrainArray, onehotTestArray, c0Train, c0Test = train_test_split(onehotArray, c0Array, test_size = testing)

    return (onehotTrainArray, onehotTestArray), (c0Train, c0Test)

def Project2Sequence(eigenvector, min = False):
    '''
    Helper function to project an eigenvector to a sequence
    Helpful when analyzing modes using PCA

    Parameters
    ----------
    eigenvector : numpy array
        A 1D array of length 200 -> used to project to a most/least cyclizable sequence

    Returns
    -------
    Sequence : str
        A projected sequence 
    '''

    assert len(eigenvector) == 200, 'eigenvector must have a length of 200'

    columnMap = {
        0 : 'A',
        1 : 'T',
        2 : 'C',
        3 : 'G'
    }

    # reshape eigenvector to 50 x 4, with each row being a 'position'
    eigenvector = eigenvector.reshape(-1, 4)

    # find the maximum value along each 'position' (row)
    if min == False:
        max = np.argmax(eigenvector, axis = 1)
    else:
        max = np.argmin(eigenvector, axis = 1)

    # generate the projected sequence
    Sequence = ''
    for columnNum in max:
        base = columnMap.get(columnNum)

        if base is None:
            raise ValueError(f"Invalid column number {columnNum} for base mapping")
        
        Sequence += base

    return Sequence

def trim(seq):
    '''
    Function to trim a sequence by removing the handles
        ! Only works if the handles are 25 bp on each side

    Paramters
    ---------
    seq: str
        string representation of a sequence 

    Returns
    -------
    seq : str
        the same string, but without any adapter sequences (total 50 bp)
    '''
    
    if len(seq) > 50:
        assert(len(seq) == 100)
        return seq[25:75]
    else:
        return seq

def envelope(x, y):
    '''
    Function cloned from the github linked in Jonghan's Research paper
    https://www.biorxiv.org/content/10.1101/2024.12.22.629997v1
    This function helps to get rid of adapter sequence effects on the variable regions cyclizability

    Paramters
    ---------
    x : list or 1D numpy array
        A list containing the indices for the c0 values input
    
    y :  list or 1D numpy array
        A list containing the c0 values for the sequences to determine the envelope for

    Returns
    -------
    U_bound : 1D numpy array
        The upper bound of the envelope for all sequences
        
    L_bound : 1D numpy array
        The lower bound of the nevelope for all sequences
    '''
    x, y = list(x), list(y)
    uidx, ux, uy = [0], [x[0]], [y[0]]
    lidx, lx, ly = [0], [x[0]], [y[0]]

    # local extremas
    for i in range(1, len(x)-1):
        if (y[i] == max(y[max(0, i-3):min(i+4, len(y))])):
            uidx.append(i)
            ux.append(x[i])
            uy.append(y[i])
        if (y[i] == min(y[max(0, i-3):min(i+4, len(y))])):
            lidx.append(i)
            lx.append(x[i])
            ly.append(y[i])

    uidx.append(len(x)-1)
    ux.append(x[-1])
    uy.append(y[-1])
    lidx.append(len(x)-1)
    lx.append(x[-1])
    ly.append(y[-1])

    ubf = interp1d(ux, uy, kind='cubic', bounds_error=False)
    lbf = interp1d(lx, ly, kind='cubic', bounds_error=False)
    U_bound = np.array([y, ubf(x)]).max(axis=0)
    L_bound = np.array([y, lbf(x)]).min(axis=0)

    return U_bound, L_bound

if __name__ == '__main__':
    pass