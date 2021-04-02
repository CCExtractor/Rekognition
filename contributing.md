# Development Enironment setup
## Install python 3.6
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```


## Clone the repository and setup venv
```
git clone https://github.com/pymit/Rekognition
cd Rekognition
./setup.sh
source myenv/bin/activate
```
***
## Postgres setup

	sudo apt update
	sudo apt install postgresql postgresql-contrib
	sudo service postgresql start
	sudo -u postgres psql

	CREATE DATABASE pmr;
	CREATE USER admin WITH PASSWORD 'admin';
	ALTER ROLE admin SET client_encoding TO 'utf8';
	ALTER ROLE admin SET default_transaction_isolation TO 'read committed';
	ALTER ROLE admin SET timezone TO 'UTC';
	ALTER USER admin CREATEDB;
	ALTER DATABASE pmr OWNER TO admin;
***

## ReactJS setup for frontend 

	git clone https://github.com/pymit/RekoUI
	cd RekoUI
	sudo apt install npm
	sudo npm install -g npm@latest
	npm install
	npm start
***

## Downloading the models
##### current directory  Rekognition

	cd media 
	mkdir {face,output,similarFace,text,object}
	cd ..
	cd corelib/model
	mkdir facenet
	cd facenet
	wget https://www.dropbox.com/s/jm8grrifh5yk7is/2017.zip?dl=1 -O 2017.zip
	unzip 2017.zip
	rm 2017.zip
	cd ..
	mkdir tfs
	cd tfs
	wget https://www.dropbox.com/s/v0ai89jj5npowt1/tfs.zip
	unzip tfs.zip
	rm tfs.zip
	cd ../../..
	cd data
	mkdir text_reco
	wget https://www.dropbox.com/s/dl/h2owqbmnrsvqo0c/ord_map_en.json
	wget https://www.dropbox.com/s/dl/yzkijd7j5yflhli/char_dict_en.json
	cd ..	

***
## TensorFlow Serving setup using Docker
	sudo apt-get update
	sudo apt install docker.io
	sudo chmod 666 /var/run/docker.sock
	docker pull tensorflow/serving:nightly-devel

` docker run -it -p 8500:8500 -p 8501:8501 -v  <absolute path to tfs model's parent directory>:/home/ tensorflow/serving:nightly-devel`

#### then in docker shell, run the below command

`tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=/home/configs/models.conf`

***

## Apply migrations 
* Migrate

    ```
    python manage.py makemigrations
    python manage.py migrate
    ```
* Staticfiles
    ```
    python manage.py collectstatic  --dry-run
    ```

## Install ffmpeg
``` 
sudo apt install ffmpeg
```

## Start django application

```
python manage.py runserver 8000
```
Django app can be accessed at http://localhost:8000

ReactJS app can be accessed at http://localhost:3000
