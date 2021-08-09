#! /bin/bash

source $PWD/myenv/bin/activate

mkdir tests/testdata
cd tests/testdata
wget https://www.dropbox.com/s/1bnxg32zvgjv0pl/compareImage.jpeg
wget https://www.dropbox.com/s/1bnxg32zvgjv0pl/compareImage.jpeg
wget https://www.dropbox.com/s/x3qpga9gc4ifamn/t1.png
wget https://www.dropbox.com/s/l5t09lp8u4ok593/t2.jpeg
wget https://www.dropbox.com/s/hzlpo74tk0xwzzh/t3.jpeg
wget https://www.dropbox.com/s/lni50cgunua5mij/test1.mp4
wget https://www.dropbox.com/s/wm3llx0ydbnq8mn/test2.mp4
wget https://www.dropbox.com/s/ato4fie6k3lmctu/test3.mp4
wget https://www.dropbox.com/s/ifd7254x29oxjze/test4.mp4
wget https://www.dropbox.com/s/iwtgwz24eipd629/obj1.mp4
wget https://www.dropbox.com/s/ull2tqlou1p8l16/obj2.mp4
wget https://www.dropbox.com/s/3w5ghr5jj6opr58/scene1.mp4
wget https://www.dropbox.com/s/ij5hj4hznczvfcw/text.mp4
cd ../..
export DJANGO_SETTINGS_MODULE="Rekognition.settings"

flake8 --exclude myenv
python -m tests.test_views
python -m tests.test_models




