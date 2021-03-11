#! /bin/bash

virtualenv -p python3.6 myenv
source $PWD/myenv/bin/activate
pip install -r requirements.txt

git clone https://github.com/Parquery/lanms
cd lanms
pyv="$(python3 -c 'import sys; a=sys.version_info;print("python"+str(a.major)+"."+str(a.minor))')"
sed -i 's/python3/'$pyv'/g' Makefile

make


sed -i 's/'$pyv'/python3/g' Makefile

cp -r $PWD/lanms $PWD/../myenv/lib/python3.6/site-packages/ 
cd ..



