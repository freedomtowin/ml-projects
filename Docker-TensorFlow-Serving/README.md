## Serve a Pretrained TensorFlow Image Classifier with Docker (locally)

Requirements: Docker, tensorflow/serving (image), python 3.8, tensorflow 2.3.0

This project loads in a pretrained model and saves it in a TF2.0 SavedModel format.

A docker image with tensorflow/serving is created. The SavedModel is copied to the image for serving.

A test image is downloaded and sent to the server for scoring using REST. The result is returned.

Here are the steps to run this project: 

0) cd to Docker-TensorFlow-Serving

1) delete the mobilenet_final folder

2) run `python save-pretrained-savedmodel.py` 

3) run the Docker commands in create_container.ps1

4) run `python call-model-server.py`