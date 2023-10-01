# surveillance camera
Keep pictures/videos only with objects of interset(e.g. human, dog, cat, car etc.) of surveillance cameras
## Why this project
For home users of surveillance cameras, normally we don't have a dedicted NVR (Network video recorder). So when we need to find something of intereset, we need browse through all the pictures/videos. This can take lots of effort. I designed this project to automatically download pictures/videos from cameras and keep the pictures/videos with objects of interest in a processed folder.
## Requirments
Only 3 third-party modules are used currently.
- piexif
- imagehash
- tqdm
## Find out your camera's sd card url
Try to use the command `curl http://admin:password@192.168.1.201:80/sd/ | html2text`, if your camera's ip address is 192.168.1.201, and port is 80. If your see a directory stucture like this. That's it.
  ![Screenshot 2023-10-01 at 18 32 29](https://github.com/Josef-Pan/surveillance_camera/assets/20598795/506f0717-1061-4f72-ac2c-63872844acad)
## Setup your camera to save pictures and videos to sd card
## Edit a cameras.ini file in the working directory and include all the cameras in it
eg.
- `http://admin:password@192.168.1.201:80/sd`
- `http://admin:password@192.168.1.202:81/sd`
- `http://admin:password@192.168.1.203:80/sd`
## Processing video files
Processing video files can take quite long time, for video files of a whole day. It may take 3 hours on Intel i7 12th generation. GPU gives little help, because most of the time is spent on removing duplicate/similar frames in the video file. A good pratice is to run this file every hour or every two hours. All processed images and videos will not be processed again.
## Adding custom classes to detect
By default, the program only detects person, dog, cat 3 classes. Actually you may add as many classes if you like only if the are supported by yolo.
You may use command `python3 surveillance_camera.py --list_classes` to see supported classes. Or you may visit YOLO official website https://docs.ultralytics.com/yolov5/ to find out all supported classes

