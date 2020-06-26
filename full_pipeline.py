# USAGE
# python full_pipeline.py --livemodel output_liveness/liveness.model --le-liveness output_liveness/le.pickle --detector face_detector  --recognizer output_recognizer/recognizer.pickle  --le-recognizer output_recognizer/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.config import list_physical_devices, experimental
import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import copy
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--livemodel", type=str, required=True, help="path to trained livness model")
ap.add_argument("-l", "--le-liveness", type=str, required=True, help="path to label encoder of liveness model")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,  help="minimum probability to filter weak detections")
ap.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
ap.add_argument("-b", "--le-recognizer", required=True, help="path to label encoder of recognizer model")
args = vars(ap.parse_args())

#only grow tensorflow memory usage as is needed the process(good for low memory GPUs)
physical_devices = list_physical_devices('GPU')
try:
    experimental.set_memory_growth(physical_devices[0], True)
    dlib.DLIB_USE_CUDA = True
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
livemodel = load_model(args["livemodel"])
le_liveness = pickle.loads(open(args["le_liveness"], "rb").read())

# load the actual face recognition model along with the label encoder
print("[INFO] loading face recognizer...")
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le_recognizer = pickle.loads(open(args["le_recognizer"], "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # pass the blob through the network and obtain the detections and
    # predictions.
    net.setInput(blob)
    detections = net.forward()

    #Initialize boxes for face recognition
    boxes = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and make a copy for passing it to 
            # face recognizer if needed
            face = frame[startY:endY, startX:endX]
            face2 = copy.deepcopy(face)

            # Preproces the face ROI in the exact same manner as our 
            # training livenss data
            try:
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
            except:
                raise Exception("Your face is too close to the camera. Please move your face slightly away from it and try again")

            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = livemodel.predict(face)[0]
            j = np.argmax(preds)
            label = le_liveness.classes_[j]

            # draw the label and bounding box on the frame if fake, else save
            # the bounding  box for face recognition. Threshold for real is 0.75
            if label == "fake" or preds[1]<0.75:
                label = "{}: {:.4f}".format("fake", preds[0])
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
            else:
                #save the bounding  box for face recognition.
                boxes.append((startY, endX, endY, startX))

    # compute the facial embedding for the faces
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names and their probabilities
    names = []
    probs = []

    # loop over the facial embeddings
    for encoding in encodings:
        # perform classification to recognize the face
        preds = recognizer.predict_proba(encoding.reshape(1,-1))[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le_recognizer.classes_[j]
        names.append(name)
        probs.append(proba)

    # loop over the recognized faces
    for ((top, right, bottom, left), name, proba) in zip(boxes, names, probs):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, "{}: {:.2f}%".format(name, proba*100), (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()