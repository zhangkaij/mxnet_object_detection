#!/bin/bash

WORKING_PATH=$(pwd);
VIRTUAL_ENV_NAME=mxnet;
export PATH=${WORKING_PATH}/${VIRTUAL_ENV_NAME}/bin:${WORKING_PATH}/${VIRTUAL_ENV_NAME}/lib:${WORKING_PATH}/${VIRTUAL_ENV_NAME}/lib/python3.6/site-packages:$PATH;
export PYTHONPATH=${WORKING_PATH}/${VIRTUAL_ENV_NAME}/lib/python3.6/site-packages:$PYTHONPATH;

python --version
echo "begin to running"
python tst.py


