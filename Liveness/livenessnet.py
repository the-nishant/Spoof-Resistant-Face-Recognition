# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import backend as K

class LivenessNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# first CONV => CONV => POOL block
		model.add(Conv2D(16, 3, padding="same", activation="relu", input_shape=inputShape))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(16, 3, padding="same", activation="relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(2))
		model.add(Dropout(0.25))

		# second CONV => CONV => POOL block
		model.add(Conv2D(32, 3, padding="same", activation="relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, 3, padding="same", activation="relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(2))
		model.add(Dropout(0.25))

		# third CONV => CONV => POOL block
		model.add(Conv2D(64, 3, padding="same", activation="relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, 3, padding="same", activation="relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(2))
		model.add(Dropout(0.25))

		# first (and only) fully connected(Dense) layer
		model.add(Flatten())
		model.add(Dense(64, activation="relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes, activation="softmax"))

		# return the constructed network architecture
		return model