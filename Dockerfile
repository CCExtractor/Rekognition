FROM python:3.6

# set work directory
WORKDIR /usr/src/PMRekognition

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install psycopg2 dependencies
# install psycopg2
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

# install dependencies
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./entrypoint.sh .

# copy project
COPY . .

# download models
RUN chmod +x download_model.sh
RUN ./download_model.sh
ENTRYPOINT ["/usr/src/PMRekognition/entrypoint.sh"]
