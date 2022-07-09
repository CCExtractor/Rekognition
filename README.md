## Poor Man's Rekognition
---
![](https://www.ccextractor.org/_media/public:gsoc:gsoc-cc.png)
Google Summer Of Code Project under CCExtractor Development

[![Build Status](https://travis-ci.org/ccextractor/Rekognition.svg?branch=master)](https://travis-ci.org/CCExtractor/Rekognition)
[![Python 3.X](https://img.shields.io/badge/python-3.X-blue.svg)](https://www.python.org/downloads/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/ccextractor/Rekognition/blob/master/LICENSE)

---
This project aims at providing a free alternative to Amazon Rekognition services. 

## Setup
### For End-User
```
git clone https://github.com/pymit/Rekognition

docker image build ./
```
Note down the IMAGEID at the end and run the docker

```
docker run -p 8000:8000 <IMAGEID>
```
### For Developers
To setup the project locally for development environment check this wiki [link](https://github.com/YB221/Rekognition/blob/master/contributing.md)


## Usage
This project currently supports
| Feature     | cURL        |
| :---        | :----       |
Face Recognition with FaceNet  |`curl -i -X POST -H "Content-Type: multipart/form-data " -F "file=@<path to image file> " --form network=1 http://127.0.0.1:8000/api/image/` | 
Face Recognition with RetinaNet  |`curl -i -X POST -H "Content-Type: multipart/form-data " -F "file=@<path to image file> " --form network=2 http://127.0.0.1:8000/api/image/`    |
| Similar Face Search   | `curl -i -X POST -H "Content-Type: multipart/form-data" -F "file=@ <path to reference image>" -F "compareImage=@ <path to compare Image>" http://127.0.0.1:8000/api/simface/`               |
| NSFW Classifier       | `curl -i -X POST -H "Content-Type: multipart/form-data " -F "file=@<path to image file> " http://127.0.0.1:8000/api/nsfw/`        |
| Text Extraction       | `curl -i -X POST -H "Content-Type: multipart/form-data " -F "file=@<path to image file> " http://127.0.0.1:8000/api/scenetext/`   |
| Object Detection      | `curl -i -X POST -H "Content-Type: multipart/form-data " -F "file=@<path to image file> " http://127.0.0.1:8000/api/objects/`     |
| Scene Classification  | `curl -i -X POST -H "Content-Type: multipart/form-data " -F "file=@<path to image file> " http://127.0.0.1:8000/api/scenedetect/` |

Details on documentation can be found [here](https://github.com/pymit/Rekognition/wiki/API-Documentation).

Currently, some features are under development which are:
- Caption generation 
- Action recognition

## Communication
Real-time communication for this project happens on slack channel of CCExtractor Development, channel [link](https://rhccgsoc15.slack.com/). You may join this channel via this [link](https://ccextractor.org/public:general:support)

## Recent GSoC archives
- ![GSoC 2021](https://summerofcode.withgoogle.com/archive/2021/organizations/5102272740065280)
- ![GSoC 2020](https://summerofcode.withgoogle.com/archive/2020/organizations/5987859833552896)
- ![GSoC 2019](https://summerofcode.withgoogle.com/archive/2019/organizations/6733668347805696)
- ![GSoC 2018](https://summerofcode.withgoogle.com/archive/2018/organizations/5152211763986432)

## References
This project uses the following.
1. [FaceNet](https://github.com/davidsandberg/facenet)
2. [CRNN](https://arxiv.org/pdf/1507.05717.pdf)
3. [EAST](https://arxiv.org/pdf/1704.03155.pdf)
4. [Synth90k](https://www.robots.ox.ac.uk/~vgg/data/text/)
5. [YOLOv3](https://pjreddie.com/darknet/yolo/)
6. [Places365](http://places2.csail.mit.edu/)
7. [RetinaFace](https://arxiv.org/pdf/1905.00641.pdf)


## License
This software is licensed under GNU GPLv3. Please see the included [License file](https://github.com/pymit/Rekognition/blob/master/LICENSE).


