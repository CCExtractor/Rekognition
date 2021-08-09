#! /bin/bash

virtualenv -p python3.6 myenv
source $PWD/myenv/bin/activate
pip install -r requirements.txt


