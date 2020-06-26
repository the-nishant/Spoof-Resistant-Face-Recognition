# USAGE
# python gather_examples.py --input liveness_videos/<video_name>.mp4 --input2 liveness_images/<folder name> --output dataset_liveness/<real or fake>/<suitable name> --output2 dataset_liveness/<real or fake>/<suitable name> --detector face_detector --skip <try 3, try to take around 150 images for 20s video> 


# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path to input video")
ap.add_argument("-j", "--input2", type=str, help="path to input images")
ap.add_argument("-o", "--output", type=str, help="path to output directory of cropped faces from video")
ap.add_argument("-p", "--output2", type=str, help="path to output directory of cropped faces from images")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16, help="number of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

if args["input"] is not None and args["output"] is not None:
	# open a pointer to the video file stream and initialize the total
	# number of frames read and saved thus far
	vs = cv2.VideoCapture(args["input"])
	read = 0
	saved = 0

	# loop over frames from the video file stream
	while True:
		# grab the frame from the file
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# increment the total number of frames read thus far
		read += 1

		# check to see if we should process this frame
		if read % args["skip"] != 0:
			continue

		# grab the frame dimensions and construct a blob from the frame
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))

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
				# the face and extract the face ROI
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]

				# write the frame to disk
				p = os.path.sep.join([args["output"],
					"{}.png".format(saved)])
				try:
					cv2.imwrite(p, face)
					saved += 1
					print("[INFO] saved {} to disk".format(p))
				except:
					print("[WARNING] no face in frame {}".format(read))				

	# do a bit of cleanup
	vs.release()
	cv2.destroyAllWindows()

if args["input2"] is not None and args["output2"] is not None:
	# grab the paths to the input images in our dataset
	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images(args["input2"]))

	#initialize the total number of images saved thus far
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
                p = os.path.sep.join([args["output2"],
                    "{}.png".format(saved)])
                try:
	                cv2.imwrite(p, face)
	                saved += 1
	                print("[INFO] saved {} to disk".format(p))
	            except:
	                print("[WARNING] no face found in {} ".format(imagePath))