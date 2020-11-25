# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

# 1.
FROM ubuntu:16.04

# 2. Defining some variables used at build time to install Python3
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.6.6

# 3. Install some handful libraries like curl, wget, git, build-essential, zlib
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        libopencv-dev \
        openssh-client \
        openssh-server \
        vim \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# 4. Installing Python3, upgrading pip and creating symbolic link for python3
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
        tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
        ./configure && make && make install && \
        apt-get update && apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
        make && make install && rm -rf ../Python-$PYTHON_VERSION* && \
        ln -s /usr/local/bin/pip3 /usr/bin/pip

RUN ${PIP} --no-cache-dir install --upgrade pip
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# FIX conflict
RUN ${PIP} install --no-cache --upgrade boto3==1.14.55

WORKDIR /

# 5. Copy our package to the WORKDIR
COPY code/custom_lightgbm_framework-1.0.0.tar.gz /custom_lightgbm_framework-1.0.0.tar.gz

# 6. Installing numpy, pandas, scikit-learn, scipy, lightgbm and our custom package
RUN ${PIP} install --no-cache --upgrade \
        numpy==1.14.5 \
        pandas==0.24.1 \
        scikit-learn==0.20.3 \
        requests==2.21.0 \
        scipy==1.2.2 \
        lightgbm==2.3.1 \
        joblib==0.14.1 \
        /custom_lightgbm_framework-1.0.0.tar.gz && \
    rm /custom_lightgbm_framework-1.0.0.tar.gz

# 7. Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 8. Set environment variable for SageMaker Training Toolkit
ENV SAGEMAKER_TRAINING_MODULE custom_lightgbm_framework.training:main
