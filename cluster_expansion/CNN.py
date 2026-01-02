'''
Functions to train the Convolutional Neural Network and Generate Sequences
'''


from .encoding import *

import numpy as np
import pandas as pd
import tensorflow as tf 
import os
from sklearn.model_selection import train_test_split

def BuildCNN(Kernel = (3,4)):
    '''
    Function to construct a Convolutional Neural Network for cyclizability prediction
    (Matches Margaritas Architecture)

    Parameters
    ----------
    Kernel : tuple ( default '(3,4)' )
        The size of the 2D convolutional kernel for (N,4) where N is the positions and 4 represents bases per position (ATCG)

    Returns
    -------
    model : tensorflow sequential model
        The framework for a convolutional neural network capable of predicting sequence cyclizability
    '''
    # initialize sequential model
    model = tf.keras.models.Sequential()

    # input layers
    model.add(tf.keras.layers.Input(shape = (200, 1)))
    model.add(tf.keras.layers.Reshape((50, 4, 1)))

    KERNEL = Kernel
    NUM_KERNELS = KERNEL[1]**3
    PAD = 'same'

    # convolutional layers
    model.add(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))
    model.add(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))
    model.add(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))

    # flattened output
    model.add(tf.keras.layers.Flatten())

    # dense layers
    model.add(tf.keras.layers.Dense(125, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))

    # output layer
    model.add(tf.keras.layers.Dense(1))

    # compile model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999),
                  loss = tf.keras.losses.MeanSquaredError(),
                  metrics = [tf.keras.metrics.MeanSquaredError()]
                 )

    return model

def formatData(seqArray, c0Array):
    '''
    Function to format the data for machine learning

    Parameters
    ----------
    seqArray : numpy or pandas array
        A 1D Numpy/Pandas array containing sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the seqArray

    Returns
    -------
    onehotArray : numpy array
        A 4D onehot array of shape (NumSequences, 50, 4, 1)

    c0Array : numpy array
        A 1D Numpy array containing cyclizability values associated to each sequence
        in the onehotArray
    '''

    onehotArray = Data2Onehot(seqArray).reshape(-1, 200, 1)
    c0Array = np.array(c0Array, dtype = np.float32).reshape(-1, 1)

    return onehotArray, c0Array

def TrainModel(dataFilepath, sep):
    '''
    Function to train the CNN model for cyclizability prediction

    Parameters
    ----------
    dataFilepath : str
        The absolute filepath to the training data

    sep : str
        The field separator in the training data

    Returns
    -------
    c0predictionModel : tensorflow sequential model
        The trained prediction model (with loaded weights)
    
    c0Predict and c0Test : numpy arrays
        Datasets containing the predicted and measured values for validating model's effectiveness
    '''

    # ensure that the data exists
    dataFilepath = os.path.abspath(dataFilepath)
    assert os.path.exists(dataFilepath), f"File not found: {dataFilepath}"

    df = pd.read_csv(dataFilepath, sep = sep)

    # extract sequence array and cyclizability array
    seqArray = df['Sequence']
    c0Array = df['C0free']

    # split formatted data into training and testing
    onehotArray, c0Array = formatData(seqArray, c0Array)

    # build the model
    c0predictionModel = BuildCNN()

    c0predictionModel.fit(onehotArray, c0Array,
                          epochs = 20,
                          validation_split = 0.1,
                          batch_size = 64,
                          verbose = 1,
                          callbacks = [
                              tf.keras.callbacks.EarlyStopping(
                                  monitor = 'val_loss',
                                  patience = 3,
                                  restore_best_weights = True
                              )
                          ])
    
    # test models predictive power on training set
    c0Predict = c0predictionModel.predict(onehotArray)
    print(f'R for the prediction model: {np.corrcoef(c0Array.T, c0Predict.T)}')

    return c0predictionModel, (c0Predict, c0Array)

def GenerateSyntheticSeqs(numSequences):

    # length of the sequence
    seqLength = 50
    # 4x4 identity matrix
    mapping = np.eye(4)

    # generating random sequences
    idx = np.random.randint(0, 4, size = (numSequences, seqLength))
    onehotArray = mapping[idx].reshape(-1, 200).astype(np.int8)
    
    return onehotArray

def GenerateSyntheticData(model, numSequences, bpDistribution = [0.25, 0.25, 0.25, 0.25]):
    '''
    Function to generate sequences

    Parameters
    ----------
    model : tensorflow sequential model
        The trained prediction model (with loaded weights)
    
    numSequences : int
        The number of sequences to generate

    Returns
    -------
    onehotArray : numpy array
        A 2D Numpy array containing onehot encoded sequences

    c0Array : numpy array
        A 1D Numpy array containing cyclizabilities associated to each sequence in seqArray
    '''
    seqLength = 50

    bp = np.asarray(bpDistribution, dtype=np.float32)
    if bp.size != 4:
        raise ValueError("bpDistribution must have length 4")
    if not np.isclose(bp.sum(), 1.0):
        raise ValueError("bpDistribution must sum to 1 (no normalization performed)")
    if (bp < 0).any():
        raise ValueError("bpDistribution must not contain negative values")


    # 4x4 identity matrix
    mapping = np.eye(4, dtype = np.int8)

    # generating random sequences
    onehotArray = np.random.choice(4, size = (numSequences, seqLength), p = bp)
    onehotArray = mapping[onehotArray]
    onehotArray = onehotArray.reshape(numSequences, seqLength * 4, 1).astype(np.float32)

    # predict cyclizabilities associated with the sequences

    c0Array = model.predict(onehotArray, batch_size = 256)

    # reshape back to usable shape
    onehotArray = onehotArray.reshape(numSequences, 200)
    c0Array = c0Array.reshape(len(c0Array))

    return onehotArray, c0Array

def YonghansArchitecture():
    '''
    Function to build and compile a model as detailed by Jonghans paper on DNA mechanics
    https://www.biorxiv.org/content/10.1101/2024.12.22.629997v1

    Parameters
    ----------
    None

    Returns
    -------
    JonghansModel : tensorflow-keras sequential model
        The compiled Keras Sequential model implementing Jonghan's architecture outlined in the linked article
    '''

    # initialize the Sequential model
    YonghansModel = tf.keras.models.Sequential()

    # input layer
    YonghansModel.add(tf.keras.layers.Input(shape = (200, 1)))

    # convolutional layers
    YonghansModel.add(tf.keras.layers.Conv1D(
        filters = 64,
        kernel_size = 28, 
        strides = 4,
        activation = tf.nn.relu,
        use_bias = True
    ))
    YonghansModel.add(tf.keras.layers.Conv1D(
        filters = 32, 
        kernel_size = 33,
        strides = 1,
        activation = tf.nn.relu,
        use_bias = True
    ))

    # flatten layer
    YonghansModel.add(tf.keras.layers.Flatten())

    # connected layer
    YonghansModel.add(tf.keras.layers.Dense(
        50,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    ))

    # output connected layer
    YonghansModel.add(tf.keras.layers.Dense(1))

    # compoile the model with optimizer, loss and metrics
    YonghansModel.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999),
                          loss = tf.keras.losses.MeanSquaredError(),
                          metrics = [tf.keras.metrics.MeanSquaredError()]
                          )
    
    return YonghansModel

def MargaritasArchitecture(Kernel = (3,4)):
    '''
    Function to build and compile a model

    Parameters
    ---------
    Kernel : tuple ( default '(3,4)' )
        The size of the 2D convolutional kernel for (N,4) where N is the positions and 4 represents bases per position (ATCG)

    Returns
    -------
    MargaritasModel : tensorflow-keras sequential model
        The compiled Keras Sequential model implementing Margaritas architecture
    '''

    # initialize the Sequential model
    MargaritasModel = tf.keras.models.Sequential()

    # input layer
    MargaritasModel.add(tf.keras.layers.Input(shape = (200, 1)))
    MargaritasModel.add(tf.keras.layers.Reshape((50, 4, 1)))

    KERNEL = Kernel # N positions, 4 bases at that position
    NUM_KERNELS = KERNEL[1]**3
    PAD = 'same' # other option is 'valid'
    # convolutional layers
    MargaritasModel.add(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))
    MargaritasModel.add(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))
    MargaritasModel.add(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))

    # flatten layer
    MargaritasModel.add(tf.keras.layers.Flatten())


    # connected layers
    MargaritasModel.add(tf.keras.layers.Dense(125, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    MargaritasModel.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
    MargaritasModel.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))

    # output connected layer
    MargaritasModel.add(tf.keras.layers.Dense(1))

    # compile the model with optimizer, loss, and metrics
    MargaritasModel.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999),
                            loss = tf.keras.losses.MeanSquaredError(),
                            metrics = [tf.keras.metrics.MeanSquaredError()]
                            )
    
    return MargaritasModel

def SymmetricArchitecture(Kernel = (7,4)):
    '''
    Builds a symmetry-enforced CNN predicting the average of a sequence and its reverse complement sequence cyclizability

    # note: fit with .fit([seq_arr, seq_rc_arr], c0, ...)
    
    Parameters
    ----------
    Kernel : tuple ( default '(3,4)' )
        The size of the 2D convolutional kernel for (N,4) where N is the positions and 4 represents bases per position (ATCG)

    Returns
    -------
    sharedNetwork : tensorflow-keras model
            The compiled keras model implementing symmetric architecture design
    '''

    # the base layers of the symmetry architecture
    def baseLayers(Kernel = Kernel):
        layers = []
        layers.append(tf.keras.layers.Reshape((50,4,1)))

        KERNEL = Kernel
        NUM_KERNELS = KERNEL[1]**3
        PAD = 'same'

        # convolutional layers
        layers.append(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))
        layers.append(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))
        layers.append(tf.keras.layers.Conv2D(NUM_KERNELS, KERNEL, activation=tf.nn.relu, kernel_initializer='he_uniform', padding=PAD))

        # flatten
        layers.append(tf.keras.layers.Flatten())

        # dense/connected layers
        layers.append(tf.keras.layers.Dense(125, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        layers.append(tf.keras.layers.Dense(25, activation=tf.nn.relu))
        layers.append(tf.keras.layers.Dense(5, activation=tf.nn.relu))

        # output layer
        layers.append(tf.keras.layers.Dense(1))

        return tf.keras.Sequential(layers)
    
    # consolidate the layers within symmetricModel
    symmetricModel = tf.keras.Sequential([tf.keras.layers.Input(shape = (200,1)), baseLayers()])

    mse = tf.keras.losses.MeanSquaredError()

    def symmetric_train_step(data):

        (seqBatch, seqRCBatch), labels = data
        with tf.GradientTape() as tape:
            # forward pass
            seq_pred = symmetricModel(seqBatch, training=True)
            seqRC_pred = symmetricModel(seqRCBatch, training=True)

            # loss on sequence predictions
            loss_seq = mse(labels, seq_pred)
            loss_seqRC = mse(labels, seqRC_pred)

            data_loss = 0.5 * (loss_seq + loss_seqRC)

            # backpropogation on 125 node dense layer
            regularizer_loss = tf.add_n(symmetricModel.losses) if symmetricModel.losses else 0.0

            # total loss as avereage of each individual loss
            total_loss = data_loss + regularizer_loss
            
        # Compute gradients and update
        gradients = tape.gradient(total_loss, symmetricModel.trainable_variables)
        symmetricModel.optimizer.apply_gradients(zip(gradients, symmetricModel.trainable_variables))

        # update metrics
        avg_pred = 0.5 * (seq_pred + seqRC_pred)
        for metric in symmetricModel.metrics:
            metric.update_state(labels, avg_pred)
            
        # return logs
        logs = {m.name: m.result() for m in symmetricModel.metrics}
        logs.update({
            "loss": total_loss,
            "loss_seq": loss_seq,
            "loss_seqRC": loss_seqRC,
            "data_loss" : data_loss,
            "regularizer_loss" : regularizer_loss
        })

        return logs
    
    symmetricModel.train_step = symmetric_train_step

    symmetricModel.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999),
        metrics = [tf.keras.metrics.MeanSquaredError(name="mse")]
    )

    return symmetricModel

if __name__ == "__main__":
    pass