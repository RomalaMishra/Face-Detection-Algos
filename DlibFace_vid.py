# Author List:		[ Romala ]

# importing the required variables
import argparse
import dlib
import cv2
import os

class FaceDetection():

	def __init__(self):

		# argument parser
		parser = argparse.ArgumentParser()
		parser.add_argument("-i", "--image", type=str, required=True,
			help="path to input image")
		self.args = vars(parser.parse_args())

		# calling the face_detect function 
		self.face_detect('/home/romala/catkin_ws/src/Ruhjaan-IITkgp/best_detected_faces')

		# skipping blank frames
		while 1:
			# read the frame
			_,image = self.cap.read()

			# calculating the fps and displaying it 
			fps = self.cap.get(cv2.CAP_PROP_FPS)
			fps = "{:.2f}".format(fps)
			image = cv2.putText(image, "fps:"+str(fps), (50,50), cv2.FONT_HERSHEY_COMPLEX, 2,(255,255,255),5)

			# to display the image window
			cv2.imshow("Output", image)
			cv2.waitKey(30)
			
		self.cap.release()


	def face_detect(self,output_directory):

		# load dlib's HOG and LinearSVM face detector
		print("loading HOG and LinearSVM face detector...")
		detector = dlib.get_frontal_face_detector()

		# to capture the video frames
		self.cap = cv2.VideoCapture(self.args["image"])
		_,image = self.cap.read()

		# converting the bgr frame to rgb
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# performing the detection
		print("performing face detection with dlib")
		results = detector(rgb)

		# to get the ROI of frames
		boxes = [self.trim_bb(image, r) for r in results]

		# to distinguish quality detected faces
		sharpness_scores = []
		for (x, y, w, h) in boxes:

			face = image[y:(y+h),x:(x+w)]
			gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			laplacian = cv2.Laplacian(gray_face, cv2.CV_64F).var()
			sharpness_scores.append(laplacian)

		# Sorting based on sharpness score
		faces = [face for _, face in sorted(zip(sharpness_scores, boxes), reverse=True)]

		# max good quality detected faces
		max_faces = 7

		# to store the quality detected faces
		for i, (x, y, w, h) in enumerate(faces[:max_faces]):
			
			# to get the ROI of detected faces and resizing to desirable size
			face = image[y:y + h, x:x + w]
			face = cv2.resize(face,(200,200))

			# storing the required faces to the output directory
			output_path = os.path.join(output_directory, f"face_{i+1}.jpg")
			cv2.imwrite(output_path, face)

	# to get the coordinates of bounding boxes of detected faces
	def trim_bb(self,image, rect):

		# to find the starting and ending x,y coordinates
		startX = rect.left()
		startY = rect.top()
		endX = rect.right()
		endY = rect.bottom()

		# to find the x,y coordinates of max face area
		startX = max(0, startX)
		startY = max(0, startY)
		endX = min(endX, image.shape[1])
		endY = min(endY, image.shape[0])

		# to find the width and height of max face area
		w = endX - startX
		h = endY - startY
		
		return (startX, startY, w, h)

if __name__ == '__main__':
    FaceDetection()



