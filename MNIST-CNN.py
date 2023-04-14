import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

mnist = keras.datasets.mnist
BATCH_SIZE = 200
EPOCHS = 3

(x_train, y_train), (x_test, y_test) = mnist.load_data() #x is the image, y is the number

#
x_train = keras.utils.normalize(x_train, axis=1).reshape(60000,28,28,1)
x_test  = keras.utils.normalize(x_test , axis=1).reshape(10000,28,28,1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

MODEL = keras.Sequential([

	#First Block
	layers.BatchNormalization(renorm=True),
	layers.Conv2D(filters=32,kernel_size=3, activation='relu',padding='same',input_shape=(28,28,1)),
	layers.Dropout(0.1),
	layers.MaxPool2D(2,2),

	#Second Block
	layers.BatchNormalization(renorm=True),
	layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
	layers.Dropout(0.1),
	layers.MaxPool2D(2,2),

	#Head Block
	layers.BatchNormalization(renorm=True),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(128, activation='relu'),
	layers.Dense(10, activation='softmax'),
])

MODEL.compile(
	optimizer = 'adam',
	loss='categorical_crossentropy',
	metrics=['accuracy'],
)

MODEL.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs = EPOCHS)

MODEL.save('MNIST-CONVOLUTIONAL')