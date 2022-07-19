Table of contents:
- [Contributing to Rekognition](#Contributing-to-Rekognition)
- [Making a PR](#making-a-pr)
- [Asking for help](#asking-for-help)
- [Development environment setup](#Development-environment-setup)

As beginners, navigating the codebase and finding your way out of the documentation can become difficult. This page will help you understand everything about contributing to howdoi and the best practices in open source as well. 

## Contributing to Rekognition
- Follow the page Setting up the development environment for setting up the development environment for Rekognition.
- Finding your first issue
- Go to issues in the Rekognition repo.
- Find the issues which you might be interested to work on. Or, you can also come up with your own ideas of improving the code.
- After finding the issue you are interested in : If the issue is an existing one, comment on the issue and ask for it to be assigned to you. Or, if the issue is unlisted and new , create a new issue and fill every information needed in the issues template provided by howdoi and ask for it to be assigned to you.
- After receiving confirmation, start working on the issue and whenever and wherever help is needed, comment on the issue itself describing your query in detail.
- A good guide on how to collaborate efficiently can be found here.

## Making a PR
- After you have worked on the issue and fixed it, we need to merge it from your forked repository into the Rekognition repository by making a PR.
- Each PR made should pass all the tests. We have new Github Actions in place for CI/CD.
- Once your commit passes all the tests, make a PR and wait for it to be reviewed and merged.


## Asking for help
- At times, help is needed while solving the issue. We recommend the following step for asking for help when you get stuck:
- Read from our documentation to see if your question has already been answered.
- Comment on the issue you are working on describing in detail what problems you are facing.
- Make sure to write your query in detail and if it is a bug, include steps to reproduce it.
- If you are not working on any issue and have a question to be answered, open a new issue on Github and wait for a reply.

# Development Environment setup
## Install python 3.6
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```

## For MacOS:
```
brew install python
```


## Clone the repository and setup venv
```
git clone https://github.com/CCExtractor/Rekognition
cd Rekognition
../setup.sh
source myenv/bin/activate
```

### For MacOS:
git clone https://github.com/CCExtractor/Rekognition
./setup.sh
cd Rekognition
python3 -m virtualenv myenv
source $PWD/myenv/bin/activate
pip install -r ../requirements.txt

NOTE: Sometimes an error "permission denied" may be shown when you try to run `setup.sh`. For this, try: `chmod 755 setup.sh` in root directory to change permissions.
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
## Postgres setup for MacOS

	brew update
	brew install postgresql
	brew services start postgresql
	psql postgres
	CREATE DATABASE pmr;
	CREATE USER admin WITH PASSWORD 'admin';
	ALTER ROLE admin SET client_encoding TO 'utf8';
	ALTER ROLE admin SET default_transaction_isolation TO 'read committed';
	ALTER ROLE admin SET timezone TO 'UTC';
	ALTER USER admin CREATEDB;
	ALTER DATABASE pmr OWNER TO admin;
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
	cd text_reco
	wget https://www.dropbox.com/s/dl/h2owqbmnrsvqo0c/ord_map_en.json
	wget https://www.dropbox.com/s/dl/yzkijd7j5yflhli/char_dict_en.json
	cd ../..	

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




