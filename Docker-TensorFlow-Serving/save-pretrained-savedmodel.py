import time
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from tensorflow.keras import layers

BATCH_SIZE = 32
IMAGE_RES = 192
    
URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/4"
mobilenet = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))


model = tf.keras.Sequential([
  mobilenet
])

model.summary()

model.compile(
  optimizer='adam', 
  loss=tf.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])

model.build()  

t = time.time()

t = time.time()

export_path_sm = "./mobilenet_final"
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

with open('input_name.txt', 'w') as file:
    file.write(model.input.name.split(':')[0])