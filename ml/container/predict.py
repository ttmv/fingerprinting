""" 
Code for making predictions for new data with previously trained model 
"""


import time
import os
import argparse
import ast
#import sagemaker
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
from scipy.fft import fft
from sagemaker.tensorflow import TensorFlowModel
from keras.models import load_model

tf.keras.backend.set_image_data_format("channels_last")

from sagemaker.estimator import Estimator
s3 = boto3.client("s3")

bucket_name = ''
subfolder = 'data00'
infofile = ""



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


def read_tf_model():
    """ 
    Read previously saved TensorFlow model from s3 bucket
    
    returns: TensorFlow model   
    """

    role = get_execution_role()
    tfmodel = TensorFlowModel(
        model_data=f"s3://{bucket_name}/model/model.tar.gz",
        role=role,
        framework_version="1.15"        
    )
    return tfmodel


def create_predictor_endpoint():
    """ 
    create new SageMaker serverless predictor endpoint for new model
    
    returns: new predictor endpoint
    """

    tfmodel = read_tf_model()
    serverless_config = ServerlessInferenceConfig()    
    predictorEp = tfmodel.deploy(serverless_inference_config=serverless_config)

    return predictorEp


def predict_new(args):
    """ 
    Creates class predictions for new event data and saves them to file on s3 bucket    
    """

    print(time.ctime(), "Creating endpoint for prediction ...")
    ep = create_predictor_endpoint()
    print(time.ctime(), "Done, reading data ...")

    # Read classes from info file:
    s3_f = s3fs.S3FileSystem()
    infof = h5py.File(s3_f.open(infofile, 'rb'), 'r')
    classes = infof["classes"]["classes"][:]
    infof.close()

    outf = open('results.csv', 'w')
    
    #read new data file names from s3 bucket:
    contents = s3.list_objects(Bucket=bucket_name, Prefix='data00/default/')['Contents']


    for cont in contents:
        if "hdf5" in cont['Key']:
            data_key = cont['Key']
            data_location = 's3://{}/{}'.format(bucket_name, data_key)    
            f = h5py.File(s3_f.open(data_location,'rb'), 'r')

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

                    # use predictor endpoint to predict class and probability
                    pred = ep.predict(X)
                    predicts = np.array(pred["predictions"])
                    comp_pred = np.argmax(predicts, axis=1)
                    
                    # save predicted class and probability to file 
                    outf.write(k + ","+ e + ',' + str(classes[comp_pred[0]]) +"," + str(comp_pred[0]) + "," + str(predicts[0][comp_pred[0]])+"\n")
            f.close()
    outf.close()
    
    # upload result file to s3 bucket: 
    s3_response = Session().upload_data("results.csv", bucket=bucket_name, key_prefix="outputs")
    
    

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'
    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    envs = dict(os.environ)
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    predict_new(parser.parse_args())
