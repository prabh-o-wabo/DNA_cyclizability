"""
Script to generate 100,000,000 Sequences (on Cluster)
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# gives the notebook access to the cluster_expansion personal library
import os
from sys import path
path.append('/Users/prabh/200 RESEARCH/Cluster Expansion Project')
import cluster_expansion as ce
from cluster_expansion import os, pd, np, plt, sns, tf, Markdown, display, train_test_split

from mpi4py import MPI

# disables tensorflow updates (it's really annoying)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### Training/loading ML Model

# adjust to actual paths
modelPath = '../../CNN_models/RC_invariant/Symmetric/S_C0freeAvg.keras'
assert os.path.exists(modelPath), 'Model path not found'

model = tf.keras.models.load_model(modelPath)

### Generating the sequences on the parallel processes

# sequences generates is a multiple of 10,000
sequences = 100_000

# 'give' each process the ML model
model = comm.bcast(model, root = 0)

# Divide work among processes
assert(sequences % size == 0), f'sequences ({sequences}) % size ({size}) = {sequences % size}; must be 0'
SeqPerProcessor = sequences // size

# generate the data in each process
localSeq, localC0 = ce.GenerateSyntheticData(model, SeqPerProcessor)

# ensure the arrays are the right type (idk chatgpt did this to fix some error)
local_seq = np.array(localSeq)
local_c0 = np.array(localC0)

# convert sequences back to 50 letter form
local_sequenceDataMl = ce.Onehot2Data(local_seq)
local_sequenceDataMl = np.array(local_sequenceDataMl) 

# gather all the sequences on the main process
allSequences = comm.gather(local_sequenceDataMl)
allC0 = comm.gather(local_c0)

# ensure the data is saved on the main process ONLY

if rank == 0:
    # concatenate to a single numpy array
    allSequences = np.concatenate(allSequences) # type: ignore
    allC0 = np.concatenate(allC0) # type: ignore

    # ensure correct dtypes
    allSequences = allSequences.astype(str)
    allC0 = allC0.astype(float)

    # stack as object array to preserve types
    saveData = np.empty((allSequences.shape[0], 2), dtype=object)
    saveData[:, 0] = allSequences
    saveData[:, 1] = allC0

    np.savetxt(f"CNN_500k.dat", saveData, delimiter='\t', fmt=('%s', '%.7f'))

