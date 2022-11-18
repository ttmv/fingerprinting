import boto3
from sagemaker.estimator import Estimator
#from sagemaker import get_execution_role
import sagemaker
import json

sm_session = sagemaker.Session(boto3.session.Session())
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

client=boto3.client('sts')
account=client.get_caller_identity()['Account']
algorithm_name="tf-rffp"
ecr_image='{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)


hyperparameters={}

def run_training():
    estimator = Estimator(
        image_uri=ecr_image,
        entry_point="rffp.py",
        role=role,
        #framework_version="2.3.1",
        model_dir="/opt/ml/model",
        #py_version="py36",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=sm_session,
        hyperparameters=hyperparameters
    )

    estimator.fit()



def run_predict():
    estimator = Estimator(
        image_uri=ecr_image,
        entry_point="predict.py",
        role=role,
        #framework_version="2.3.1",
        model_dir="/opt/ml/model",
        #py_version="py36",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=sm_session,
        hyperparameters=hyperparameters
    )

    estimator.fit()


