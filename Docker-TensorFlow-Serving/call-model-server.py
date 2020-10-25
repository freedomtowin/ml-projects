import time
import numpy as np
import matplotlib.pylab as plt
import PIL.Image as Image

import requests
import json
import cv2

import io

IMAGE_RES = 192


with open('input_name.txt', 'r') as file:
    input_name = file.read()

print('input name',input_name)

URL = "http://127.0.0.1:8501/v1/models/mobilenet_final/versions/001:predict" 

image_file = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'

req = requests.get(image_file)
image = cv2.cvtColor(cv2.imdecode(np.frombuffer(req.content, np.uint8),-1),cv2.COLOR_BGR2RGB)/255.0

headers = {"content-type": "application/json"}
image_content = cv2.resize(image,(IMAGE_RES, IMAGE_RES)).astype('uint8').tolist()
body = {"instances": [{input_name: image_content}]}
r = requests.post(URL, data=json.dumps(body), headers = headers) 

preds = json.loads(r.text)['predictions']

predicted_class = np.argmax(preds, axis=-1)

labels_file = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'

req = requests.get(labels_file)

labels = np.array(str(req.content.decode('utf-8')).splitlines())

plt.imshow(image)
plt.axis('off')
predicted_class_name = labels[predicted_class[0]]
_ = plt.title("Prediction: " + predicted_class_name)
plt.show()
