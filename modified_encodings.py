# Author List:		[ Romala ]

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-b", "--batch-size", type=int, default=3,
    help="batch size for processing images")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
totalImages = len(imagePaths)
processedImages = 0

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# process the dataset in batches
for batchStart in range(0, totalImages, args["batch_size"]):
    # extract the batch of images and paths
    batchPaths = imagePaths[batchStart:batchStart+args["batch_size"]]
    batchImages = []
    batchNames = []

    # loop over the batch of images
    for imagePath in batchPaths:
        try:
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batchImages.append(rgb)
            batchNames.append(name)
        except Exception as e:
            print(f"[INFO] Failed to process image: {imagePath}")
            print(f"[INFO] Error: {e}")

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input images
    batchBoxes = [face_recognition.face_locations(image, model=args["detection_method"]) for image in batchImages]

    # compute the facial embeddings for the faces
    batchEncodings = [face_recognition.face_encodings(image, boxes) for image, boxes in zip(batchImages, batchBoxes)]

    # loop over the encodings
    for (encoding, name) in zip(batchEncodings, batchNames):
        # add each encoding + name to our set of known names and encodings
        knownEncodings.extend(encoding)
        knownNames.extend([name] * len(encoding))

    # update the number of processed images
    processedImages += len(batchPaths)
    print("[INFO] processed {}/{} images".format(processedImages, totalImages))

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))
