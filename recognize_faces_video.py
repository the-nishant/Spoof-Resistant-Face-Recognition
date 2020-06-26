# USAGE
# python recognize_faces_video.py --recognizer output/recognizer.pickle --le output/le.pickle --detector face_detector 
# python recognize_faces_video.py --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detector", type=str, required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading face recognizer...")
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
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

	        boxes.append((startY, endX, endY, startX))

	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	probs = []

	# loop over the facial embeddings
	for encoding in encodings:
		# perform classification to recognize the face
		preds = recognizer.predict_proba(encoding.reshape(1,-1))[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]
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

	# update the FPS counter
	fps.update()

	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces t odisk
	if writer is not None:
		writer.write(frame)


	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()