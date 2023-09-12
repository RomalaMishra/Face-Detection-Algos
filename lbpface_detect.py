# Author List:		[ Romala ]

# import the necessary packages
import cv2
import argparse
import os

cascade_path = '/home/romala/catkin_ws/src/Detection/lbpcascade_frontalface.xml'
output_directory = '/home/romala/catkin_ws/src/Detection/best_detected_faces'
face_cascade = cv2.CascadeClassifier(cascade_path)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for i, (x, y, w, h) in enumerate(faces):
		
		face = image[y:(y+h),x:(x+w)]     
		output_path = os.path.join(output_directory, f"face{i+1}.jpg")
		cv2.imwrite(output_path,face)     
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)        

cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
