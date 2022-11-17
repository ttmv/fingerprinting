""" 
Code for training
"""


import time
import os
import argparse
import ast
from sagemaker import get_execution_role
from sagemaker.session import Session
import boto3
import s3fs

from tensorflow import keras
import numpy as np
import pandas as pd
import h5py
import tarfile
from sagemaker.serverless import ServerlessInferenceConfig


from keras.optimizers import gradient_descent_v2
from keras.models import Sequential, Model 
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.fft import fft
from sagemaker.tensorflow import TensorFlowModel
from keras.models import load_model

tf.keras.backend.set_image_data_format("channels_last")

from sagemaker.estimator import Estimator

s3 = boto3.client("s3")

bucket_name = 'arf-rffp1'
subfolder = 'data00'
infofile = "s3://arf-rffp1/data00/info.h5"
checkpoint_filepath = "/opt/ml/output/"
s3_f = s3fs.S3FileSystem()



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


    Session().upload_data(filename_conf1, bucket=bucket_name, key_prefix="plots")
    Session().upload_data(filename_conf2, bucket=bucket_name, key_prefix="plots")
    Session().upload_data(filename_loss, bucket=bucket_name, key_prefix="plots")
    Session().upload_data(filename_acc, bucket=bucket_name, key_prefix="plots")



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

    # Retrieve file names for training data from s3 bucket 
    # Data has previously been read from event files and saved to hdf5 files    
    contents = s3.list_objects(Bucket=bucket_name, Prefix='data00/train/')['Contents']

    for cont in contents:
        if "hdf5" in cont['Key']:
            data_key = cont['Key']
            data_location = 's3://{}/{}'.format(bucket_name, data_key)    
            f = h5py.File(s3_f.open(data_location,'rb'), 'r')
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

    

def create_model(X_train, y_train, X_val, y_val, class_amount, epochs=100):
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



def _train(args):
    """ 
    Main function for training
    """

    #prepare file system for reading from AWS s3 bucket and read classes used for training from file: 
    s3_f = s3fs.S3FileSystem()
    infof = h5py.File(s3_f.open(infofile, 'rb'), 'r')
    classes = infof["classes"]["classes"][:]
    class_amount = len(classes)
    
    
    X_train, y_train, X_test, y_test = create_train_and_test_data(classes)
    infof.close()
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True) 
    model, history = create_model(X_train, y_train, X_val, y_val, class_amount)

    model.load_weights(checkpoint_filepath)
    pred = model.predict(X_test)

    plot_results(pred, y_test, history)
    _save_model(model)

    
def _save_model(model):
    """ 
    Save model to s3 bucket 
        Parameters:
            model: trained model to save     
    """

    model.save("export/Servo/1")
    with tarfile.open("model_tmp.tar.gz", "w:gz") as tar:
        tar.add("export")
    s3_response = Session().upload_data("model_tmp.tar.gz", bucket=bucket_name, key_prefix="model")

    

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'
    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of total epochs to run (default: 5)')
    #parser.add_argument('--batch-size', type=int, default=4, metavar='BS',
    #                    help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    envs = dict(os.environ)

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    #parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())

