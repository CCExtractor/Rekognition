#!/usr/bin/env sh
rm cceface/migrations/*.py
rm cceface/migrations/*.pyc
touch cceface/migrations/__init__.py
python manage.py makemigrations
python manage.py migrate
