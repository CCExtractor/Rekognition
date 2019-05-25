# Rekognition
Poor Man's Rekognition

[![Build Status](https://travis-ci.org/pymit/Rekognition.svg?branch=master)](https://travis-ci.org/pymit/Rekognition)
[![Coverage Status](https://coveralls.io/repos/github/pymit/Rekognition/badge.svg?branch=master)](https://coveralls.io/github/pymit/Rekognition?branch=master)

Python3 is required
Setting up the project locally

Run the following command to setup the virtualenv.
```
pip3 install virtualenv
virtualenv -p python3 myenv  
source myenv/bin/activate
```
Clone the repo
```
git clone https://github.com/pymit/Rekognition
cd Rekognition
pip3 install -r requirements.txt
```


* Setup postgres database
	* Start postgresql by typing ```sudo service postgresql start```
	* Now login as user postgres by running ```sudo -u postgres psql``` and type the commands below:

        ```
        CREATE DATABASE pmr;
        CREATE USER admin WITH PASSWORD 'admin';
        ALTER ROLE admin SET client_encoding TO 'utf8';
        ALTER ROLE admin SET default_transaction_isolation TO 'read committed';
        ALTER ROLE admin SET timezone TO 'UTC';
        ALTER USER admin CREATEDB;
        ```

    * Exit psql by typing in \q and hitting enter.

* Migrate

    ```
    python manage.py makemigrations
    python manage.py migrate
    ```


Start django application

```
python manage.py runserver 8888
```
can be accessed at http://localhost:8000



## License
This software is licensed under GNU GPLv3. Please see the included License file.