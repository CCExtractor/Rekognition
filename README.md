## Poor Man's Rekognition
---
![](https://www.ccextractor.org/_media/public:gsoc:gsoc-cc.png)
Google Summer Of Code 2019 Project under CCExtractor Development

[![Build Status](https://travis-ci.org/pymit/Rekognition.svg?branch=master)](https://travis-ci.org/pymit/Rekognition)
[![Coverage Status](https://coveralls.io/repos/github/pymit/Rekognition/badge.svg?branch=master)](https://coveralls.io/github/pymit/Rekognition?branch=master)
[![Python 3.X](https://img.shields.io/badge/python-3.X-blue.svg)](https://www.python.org/downloads/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/pymit/Rekognition/blob/master/LICENSE)

---


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
        ALTER DATABASE pmr OWNER TO admin;
        ```

    * Exit psql by typing in \q and hitting enter.

* Migrate

    ```
    python manage.py makemigrations
    python manage.py migrate
    ```
* Staticfiles
    ```
    python manage.py collectstatic  --dry-run
    ```

Start django application

```
python manage.py runserver 8000
```
Django app can be accessed at http://localhost:8000


---
## License
This software is licensed under GNU GPLv3. Please see the included License file.
