# Author List:		[ Romala ]

# importing the required variables
import cv2
import mediapipe as mp

class FaceDetection():

    def __init__(self):
        
        # capturing the video
        cap = cv2.VideoCapture('/home/romala/catkin_ws/src/Ruhjaan-IITkgp/video1.mp4')

        # defining the holistic model
        holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # to skip blank frames
        while cap.isOpened():

            # reading the frames
            _,frame = cap.read()

            # converting from bgr to rgb
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            #processing the model   
            image,results = self.face_detection(frame,holistic_model)

            # resizing the img to fit the screen 
            image = cv2.resize(image,(500,500))

            # converting back from rgb to bgr
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            # drawing the face landmarks
            self.draw_landmarks(image, results)

            # displaying the output
            cv2.imshow('output', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()   
  
    def face_detection(self, image, model):
       
        # processing the model
        results = model.process(image)                 
        return image, results
      
    def draw_landmarks(self, image, results):
    
        # drawing the face landmarks
        mp.solutions.drawing_utils.draw_landmarks(
        image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS,
        mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)) 
    
if __name__ == '__main__':
    FaceDetection()