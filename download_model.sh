#!/bin/sh
mkdir -p corelib/model/{facenet,tfs}
wget https://www.dropbox.com/s/jm8grrifh5yk7is/2017.zip?dl=1  -O 2017.zip
wget https://www.dropbox.com/s/zzjzdvx6523am20/fer2013.zip?dl=1 -O fer2013.zip
unzip 2017.zip -d corelib/model/facenet/
unzip fer2013.zip -d corelib/model/tfs/
mkdir -p media/{face,images,logs,output,similarFace,videos}
rm -rf 2017.zip
rm -rf fer2013.zip