import numpy as np
import pickle
import cv2, os
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(glob('gestures/*'))

image_x, image_y = get_image_size()
print(get_image_size())
print(get_num_of_classes())
def cnn_model():
    num_of_classes = get_num_of_classes()
    model = keras.Sequential()
    model.add(layers.Conv2D(16,(2,2), input_shape=(image_x,image_y,1), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(layers.Conv2D(32,(3,3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(3,3),padding='same'))
    model.add(layers.Conv2D(64,(5,5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_of_classes,activation="softmax"))
    model.summary()
    model.compile(
        optimizer = keras.optimizers.SGD(learning_rate=0.01),
        loss = keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    filepath="cnn_model_keras2.h5"
    checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='val_accuracy',
        verbose=1, save_best_only=True, mode='max'
    )
    callbacks_list = [checkpoint1]
    tf.keras.utils.plot_model(model=model, to_file="model.png", show_shapes=True)
    return model,callbacks_list  
cnn_model()
with open("train_images.pkl", "rb") as f:
    train_images = np.array(pickle.load(f))
with open("train_labels.pkl", "rb") as f:
    train_labels = np.array(pickle.load(f), dtype=np.int32)

with open("val_images.pkl", "rb") as f:
    val_images = np.array(pickle.load(f))
with open("val_labels.pkl", "rb") as f:
    val_labels = np.array(pickle.load(f), dtype=np.int32)
print(val_labels.shape)
print(train_labels.shape)
print(val_images.shape)
print(train_images.shape)
print("change")
train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
train_labels = tf.keras.utils.to_categorical(train_labels)
val_labels = tf.keras.utils.to_categorical(val_labels)
    
print(val_labels.shape)
print(train_labels.shape)
print(val_images.shape)
print(train_images.shape)
model, callbacks_list = cnn_model()
model.summary()
model.fit(train_images, train_labels, 
    validation_data=(val_images, val_labels), 
    epochs=15, batch_size=500, callbacks=callbacks_list)
scores = model.evaluate(val_images, val_labels, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
    

tf.keras.backend.clear_session();