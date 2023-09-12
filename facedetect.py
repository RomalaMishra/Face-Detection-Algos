# Author List:		[ Romala ]

# import the necessary packages
import cv2 

class FaceDetect():

    def __init__(self):
            
        # capture frames from a webcam
        self.cap = cv2.VideoCapture('/home/romala/catkin_ws/src/Detection/video1.mp4')
        
        # loop runs if capturing has been initialized.
        while self.cap.isOpened(): 
        
            # reads frames from a camera
            _, img = self.cap.read() 
            self.detect(img)
        
        # Close the window
        self.cap.release()
        
        # De-allocate any associated memory usage
        cv2.destroyAllWindows() 

    def detect(self,img):

        # define face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # convert to gray scale of each frames
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detects faces of different sizes in the input image
        faces = face_cascade.detectMultiScale(gray)

        # To draw a rectangle in a face 
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 

        # Display an image in a window
        cv2.imshow('face detected frame',img)
        cv2.waitKey(1)

if __name__ == '__main__':
    FaceDetect()