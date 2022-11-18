import time
import os
import keras
import numpy as np
import pandas as pd
import h5py
import tarfile
from keras.optimizers import gradient_descent_v2
from keras.models import Sequential, Model 
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.fft import fft
from keras.models import load_model


infofile = "info.h5"
train_data_dir = "data00/train/"
pred_data_dir = "data00/default/"
checkpoint_filepath = "checkpoints"


#print("keras: ", keras.__version__)
#print("tf:", tf.__version__)

def plot_results(pred, y_test, history, classes):
    """ 
    Plot results for evaluating model performance
    Parameters:
      pred: model prediction
      y_test: test labels
      history: training history
      classes: array of classes
    """

    comp_pred = np.argmax(pred, axis=1)
    comp_test = np.argmax(y_test, axis=1)
    conf1 = confusion_matrix(comp_test, comp_pred) #test run
    conf2 = confusion_matrix(comp_test, comp_pred, normalize='true') #test run
    labels = classes        
    tm = time.localtime()
    timestr = str(tm.tm_year) + str(tm.tm_mon) + str(tm.tm_mday) + "-"+ str(tm.tm_hour) + ":"+ str(tm.tm_min)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf1, display_labels=labels)
    plt.rc('xtick', labelsize=11) 
    plt.rc('ytick', labelsize=11) 
    plt.rc('axes', labelsize=13) 
    fig, ax = plt.subplots(figsize=(17,17))
    disp.plot(cmap='binary', ax=ax, values_format='.2f')

    #disp.plot(cmap='Reds', values_format='')
    filename_conf1 = timestr+"-conf-abs.png" 
    plt.savefig(filename_conf1)
    plt.clf()
    

    disp = ConfusionMatrixDisplay(confusion_matrix=conf2, display_labels=labels)
    plt.rc('xtick', labelsize=11) 
    plt.rc('ytick', labelsize=11) 
    plt.rc('axes', labelsize=13) 
    fig, ax = plt.subplots(figsize=(17,17))
    disp.plot(cmap='binary', ax=ax, values_format='.2f')

    filename_conf2 = timestr+"-conf-rel.png"
    plt.savefig(filename_conf2)
    plt.clf()


    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.ylim((0, 1.2))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    filename_loss = timestr+"-loss.png"
    plt.savefig(filename_loss)
    plt.clf()


    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim((0, 1.2))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    filename_acc = timestr+"-acc.png"
    plt.savefig(filename_acc)
    plt.clf()


def pre_fft(arr):
    """ 
    Preprocessing: FFT 
    Tansforms the time domain data to frequency domain using FFT 
    
    Parameters:
        arr: time domain data array

    Returns: 
        X_arr: FFT data array
    """

    ind = 0    
    fftsize = 512*16
    
    X_arr = []
    fft_arr = fft(arr[0, :fftsize])
    fftlen = len(np.abs(fft_arr[:fftsize]))
    fftpart = fftlen//10
    maxval = np.amax(np.abs(fft_arr[:fftsize//2]))
    fft_arr = np.abs(fft_arr[:fftsize//2]) / maxval
    
    while(ind+fftpart < fftlen):
        tmp_arr = np.abs(fft_arr[:fftpart//2])
        X_arr.append(tmp_arr)
        ind = ind + fftpart // 2

    return X_arr    


def create_data_arr(data, X, Y, cur_class, N, classes):
    """
    Preprocessing: scales data, calls FFT function and forms data for the correct input format 
    parameters:
        data: data
        X: array for preprocessed data
        Y: array for class labels 
        curr_class: current class 
        N: amount of event parts to use for training
        classes: all classes as array 
    
    Returns: 
        X: array of preprocessed data
        Y: array of class labels 
    """
    
    arr_len = N*238832
    arr_sc = data*0.1
    new_arr = np.zeros((2, arr_len))
    new_arr = arr_sc[:,:arr_len]
    
    for i in range(0, N):
        x1 = pre_fft(new_arr[:,i*238832:(i+1)*238832])
        X.append(x1)
        y1 = [0]*len(classes)
        y1[classes.index(cur_class)] = 1
        Y.append(y1)
    
    return X, Y



def create_train_and_test_data(classes, nsize=27):
    """ 
    Reads data from file and splits it to train and test data

    Parameters:
        classes: classes used for training

        nsize: amount of parts from events to use for training 

    Returns:
        X_train, Y_train: data and labels for training 
        X_test, Y_test: data and labels for testing
    """

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    N=nsize
    arr_len = N*238832

    # Retrieve file names for training data from default training data directory 
    # Data has previously been read from event files and saved to hdf5 files    

    datapath = os.getcwd() + "/" + train_data_dir
    print("data in:", datapath)
    datafiles = os.listdir(datapath)

    for dfn in datafiles:
        print("cont", datapath+dfn)
        if "hdf5" in dfn:
            f = h5py.File(datapath+dfn, 'r')
            for c in classes:
                if c in f.keys():
                    grp = f[c]
                    #train data:        
                    for k in list(grp.keys())[:-2]:
                        arr = f[c][k][:]
                        create_data_arr(arr, X_train, Y_train, c, N, classes)
                    for k in list(grp.keys())[-2:]:
                        arr = f[c][k][:]
                        create_data_arr(arr, X_test, Y_test, c, N, classes)
            f.close()
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    return X_train, Y_train, X_test, Y_test


def create_model(X_train, y_train, X_val, y_val, class_amount, epochs=50):
    """ 
    LSTM model 
    Parameters:
        X_train: preprocessed training data
        y_train: classes for training data
        X_val: preprocessed validation data
        y_val: classes for validation data
        class_amount: number of classes
        epochs: number of epochs
  
    Returns:
        model: the trained model
        history: model training history      
    """

    model = Sequential()
    f_dims = X_train.shape[1:]
    model.add(InputLayer(input_shape=f_dims))
    model.add(LSTM(512, activation="relu", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation="relu"))
    model.add(Dense(64, activation='tanh', kernel_regularizer='l1')) #alkup.
    model.add(Dense(class_amount, activation='softmax'))
    
    model.summary()
    opt = gradient_descent_v2.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    earlystop = EarlyStopping(monitor='val_accuracy', patience=14)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True
    )

    callbacks = [earlystop, model_checkpoint_callback]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[callbacks]) 

    return model, history    

""" TRAINING """

def _train():
    """ 
    Main function for training
    """
    
    #read classes from infofile and transform to string array:
    infof = h5py.File(infofile, 'r')
    cl = infof["classes"]["classes"][:]
    infof.close()

    classes = []
    for c in cl:
        classes.append(c.decode('utf-8'))
    class_amount = len(classes)
    
    X_train, y_train, X_test, y_test = create_train_and_test_data(classes)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True) 
    model, history = create_model(X_train, y_train, X_val, y_val, class_amount)

    model.load_weights(checkpoint_filepath)
    pred = model.predict(X_test)

    plot_results(pred, y_test, history, classes)
    _save_model(model)

    
def _save_model(model):
    """ 
    Save model 
        Parameters:
            model: trained model to save     
    """

    model.save("export/Servo/1")
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("export")



""" PREDICTING """

def predict_new():
    """ 
    Creates class predictions for new event data and saves them to file    
    """

    model = keras.models.load_model("export/Servo/1")

    model.summary()
    
    # Read classes from info file:
    infof = h5py.File(infofile, 'r')
    cl = infof["classes"]["classes"][:]
    infof.close()

    classes = []
    for c in cl:
        classes.append(c.decode('utf-8'))
    class_amount = len(classes)
    
    outf = open('results.csv', 'a')

    datapath = os.getcwd() + "/" + pred_data_dir
    print("data in:", datapath)
    datafiles = os.listdir(datapath)

    for dfn in datafiles:
        print("cont", datapath+dfn)
        if "hdf5" in dfn:
            f = h5py.File(datapath+dfn, 'r')
            for k in f.keys():       
                evnames = f[k].keys()
                for e in evnames:

                    # read data from file and preprocess it: 
                    arr = f[k][e][:]
                    arr = np.array(arr)
                    arr_len = 238832
                    new_arr = np.zeros((2, arr_len))
                    new_arr = arr[:,:arr_len]

                    x1 = pre_fft(new_arr)
                    X = []
                    X.append(x1)
                    X = np.asarray(X)

                    # use model to predict class and probability
                    pred = model.predict(X)
                    comp_pred = np.argmax(pred, axis=1)

                    # save predicted class and probability to file 
                    outf.write(k + ","+ e + ',' + str(classes[comp_pred[0]]) +"," + str(comp_pred[0]) + "," + str(pred[0][comp_pred[0]])+"\n")

            f.close()
    outf.close()


if __name__ == '__main__':
  _train()
  #predict_new()


