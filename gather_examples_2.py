
# USAGE
# python gather_examples_2.py --input dataset_liveness_small/training_real --output dataset/real/photoshop --detector face_detector 
# python gather_examples.py --input videos/<your name>_fake.mp4 --output dataset/fake/<your name> --detector face_detector 

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
    help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
    help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["input"]))

saved = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image 
    image = cv2.imread(imagePath)

     # grab the frame dimensions and convert it to a blob
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]

            # write the frame to disk
            try:
                p = os.path.sep.join([args["output"],
                    "{}.png".format(saved)])
                cv2.imwrite(p, face)
                saved += 1
                print("[INFO] saved {} to disk".format(p))
            except:
                continue

