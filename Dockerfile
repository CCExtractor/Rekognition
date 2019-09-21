FROM python:3.6

# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

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
RUN pip install -r requirements.txt


# copy project
COPY . .

# add and run as non-root user
# RUN adduser -D myuser
# USER myuser

# run gunicorn
CMD gunicorn Rekognition.wsgi:application --bind 0.0.0.0:$PORT