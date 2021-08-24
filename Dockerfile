FROM python:3.8
RUN echo $(pwd)

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
        python3-setuptools \
        python3-pip \
        python3-dev \
        python3-venv \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get install tensorflow-model-server

# install dependencies
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

#download models and set-up folders
CMD cd media 
CMD mkdir {face,output,similarFace,text,object}
CMD cd ..
CMD cd corelib/model
CMD mkdir facenet
CMD cd facenet
CMD wget https://www.dropbox.com/s/jm8grrifh5yk7is/2017.zip?dl=1 -O 2017.zip
CMD unzip 2017.zip
CMD rm 2017.zip
CMD cd ..
CMD mkdir tfs
CMD cd tfs
CMD wget https://www.dropbox.com/s/v0ai89jj5npowt1/tfs.zip
CMD unzip tfs.zip
CMD rm tfs.zip
CMD cd ../../..
CMD cd data
CMD mkdir text_reco
CMD cd text_reco
CMD wget https://www.dropbox.com/s/dl/h2owqbmnrsvqo0c/ord_map_en.json
CMD wget https://www.dropbox.com/s/dl/yzkijd7j5yflhli/char_dict_en.json
CMD cd ../..

COPY . .

CMD tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=$(pwd)/corelib/model/tfs/model_volume/configs/models.conf

CMD python3 manage.py flush --no-input
CMD python3 manage.py migrate
CMD python3 manage.py runserver 8000
EXPOSE 8000
