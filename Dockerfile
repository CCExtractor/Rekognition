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
RUN apt-get install tensorflow_model_server

# install dependencies
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=$(pwd)/corelib/model/tfs/model_volume/configs/models.conf

CMD python3 manage.py flush --no-input
CMD python3 manage.py migrate
CMD python3 manage.py runserver 8000
EXPOSE 8000
