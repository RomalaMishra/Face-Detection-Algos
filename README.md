# Face Detection Algorithms
## Some of the very most accurate face detection algorithms

In order to use this package, go through the following steps:
Clone the repo to your files
'''bash
git clone https://github.com/RomalaMishra/Face-Detection-Algos.git
'''

### (1) Face Detection using dlib library
For images
'''bash
python3 DlibFace_img.py
'''

For videos
'''bash
python3 DlibFace_vid.py
'''

This script takes an input image/video, detects faces in it using dlib library, assesses the quality of the detected faces based on sharpness, and saves the top-quality faces to an output directory while displaying bounding rectangles around the detected faces in the original image.

### (2) Face Detection and recognition using face_recognition library
'''bash
python3 encode_face.py
'''
then run
'''bash
python3 recognizeface_img.py
'''

This script is used to process a directory of face images, detect faces in each image, compute facial encodings for these faces, and serialize the collected data into a file for later recognition or identification tasks.

### (3) Face and eye Detection using haarcascades
'''bash
python3 face_eye.py
'''

This script provides a simple real-time face and eye detection application using Haar Cascade Classifiers. It captures video from the default camera, processes each frame to detect faces and eyes, and displays the annotated video with rectangles around the detected regions

### (4) Face and eye Detection using lbpcascade
'''bash
python3 lbpface_detect.py
'''
This script performs face detection in an input image using a LBP_frontalface Cascade classifier and saves the detected faces as separate image files in an output directory.
Download the "lbpcascade_frontalface.xml" file before running this script.

### (5) Face Detection using mediapipe
'''bash
python3 mp_vid.py
'''
This script uses the MediaPipe framework to perform real-time face detection and facial landmark detection on frames from a video stream. Detected facial landmarks are overlaid on the frames, and the processed video is displayed.

## Contributions
Contributions to this project are welcome. If you have any ideas for improvements, bug fixes, or additional features, please feel free to fork the repository and submit a pull request.