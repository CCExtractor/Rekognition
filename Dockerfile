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
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" |  tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg |  apt-key add -
RUN apt-get update
RUN /usr/local/bin/python -m pip install --upgrade pip

RUN apt-get install tensorflow-model-server
WORKDIR $(pwd)

#download models and set-up folders
WORKDIR media 
RUN mkdir {face,output,similarFace,text,object}
WORKDIR ..
WORKDIR corelib/model
RUN mkdir facenet
WORKDIR facenet
RUN wget https://www.dropbox.com/s/jm8grrifh5yk7is/2017.zip?dl=1 -O 2017.zip
RUN unzip 2017.zip
RUN rm 2017.zip
WORKDIR ..
RUN mkdir tfs
WORKDIR tfs
RUN wget https://www.dropbox.com/s/v0ai89jj5npowt1/tfs.zip
RUN unzip tfs.zip
RUN rm tfs.zip
WORKDIR ../../..
WORKDIR data
RUN mkdir text_reco
WORKDIR text_reco
RUN wget https://github.com/MaybeShewill-CV/CRNN_Tensorflow/blob/master/data/char_dict/ord_map_en.json
RUN wget https://github.com/MaybeShewill-CV/CRNN_Tensorflow/blob/master/data/char_dict/char_dict_en.json
WORKDIR ../..
# install dependencies
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install -U numpy
COPY . .

RUN tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=$(pwd)/corelib/model/tfs/model_volume/configs/models.conf

RUN python3 manage.py flush --no-input
RUN python3 manage.py migrate
RUN python3 manage.py runserver 8000
EXPOSE 8000
