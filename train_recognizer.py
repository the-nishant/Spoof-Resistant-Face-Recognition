# USAGE
# python train_recognizer.py --encodings output_recognizer/encodings.pickle --recognizer output_recognizer/recognizer.pickle --le output_recognizer/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized database of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face emcodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

#convert the encodings in a numpy array and reshape them to feed into our classifier
data["encodings"] = np.array(data["encodings"]).reshape(-1,128)

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["encodings"], labels)

# evaluate the network
print("[INFO] evaluating network...")
predictions = recognizer.predict(data["encodings"])
print(classification_report(labels,
    predictions, target_names=le.classes_))

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()