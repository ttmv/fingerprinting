# SageMaker image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.0-cpu-py39-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY rffp.py /opt/ml/code/rffp.py
COPY predict.py /opt/ml/code/predict.py

# Defines rffp.py as script entrypoint 
ENV SAGEMAKER_PROGRAM rffp.py

RUN apt-get update && \
    apt-get install -y nginx

RUN pip install gevent gunicorn flask
