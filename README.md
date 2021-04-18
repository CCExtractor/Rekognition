## Poor Man's Rekognition
---
![](https://www.ccextractor.org/_media/public:gsoc:gsoc-cc.png)
Google Summer Of Code Project under CCExtractor Development

[![Build Status](https://travis-ci.org/pymit/Rekognition.svg?branch=master)](https://travis-ci.org/pymit/Rekognition)
[![Python 3.X](https://img.shields.io/badge/python-3.X-blue.svg)](https://www.python.org/downloads/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/pymit/Rekognition/blob/master/LICENSE)

---
This project aims at providing a free alternative to Amazon Rekognition services.
The current version of the application supports the following features:-
Face Recognition: The system uses biometrics to map facial features and compares that information with the database of known faces to find a match and hence identifying the identity of an individual using their face.
Similar Face Search: The feature provides a high speed search of the input image from the stored dataset based on the facial features.
NSFW Classifier: Not-Safe-For-Work(NSFW) works by filtering the not suitable images by classifying them on the predefined categories like violence, etc.
Text Extraction: The feature extracts textual data from a digital image and converts it to it’s ASCII character that a computer can recognize.
Object Detection: The feature detects, locate and trace the object from the input image.
Scene Classification: This feature classifies a scene image to one of the predefined scene categories by comprehending the entire image.
The application also supports analysis of Video data in the following features:
 1) Text detection in videos
 2) Scene classification in videos
 3) NSFW classification in videos
 4) VideoFR
 5) Object detection in videos
 The results of the above features are similar to that in the case of Images since the model applies similar algorithms in the frames of the Video.


## Setup
To setup the project locally for development environment check this wiki [link](https://github.com/pymit/Rekognition/wiki/Project-Setup-in-Ubuntu-18.04)


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


## Communication
Real-time communication for this project happens on slack channel of CCExtractor Development, channel [link](https://rhccgsoc15.slack.com/). You may join this channel via this [link](https://ccextractor.org/public:general:support)


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


