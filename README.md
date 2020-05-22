# ml-projects

## Kaggle Competition - Categorial Feature Encoding Challenge 2

#### cat-dat-ii.ipynb (tensorflow v2)

Machine learning workflow for Kaggle Competition: Categorical Feature Encoding Challenge II. The workflow explores using mean-value encoding, noise reduction techniques, and a categorical embedding layer with tensorflow. 

Private leaderboard score (AUC): 0.78685

Rank 1 private leaderboard score (AUC): 0.78820

Link: https://www.kaggle.com/c/cat-in-the-dat-ii/

## Deep Autoregressive Models, Forecasting with TensorFlow 2.0

#### deep-autoregressive-model-tensorflow.ipynb (tensorflow v2)

A formulation for deep autoregressive models was created using the 1st order approximation of the autoregressive function. A deep autoregressive model was created, in TensorFlow, to forecast temperature and rainfall in Melbourne AU. The forecasts were created by snowballing the predictions back into the model. The result showed that the model was capable of capturing small global trends and periodic patterns.

Blog Post: https://freedomtowin.github.io/2020/05/12/Deep-Autoregressive-Models.html

## 1-D Covolutional Network, Tensorflow 1.0

#### [https://github.com/freedomtowin/ml-projects/blob/master/1d-convolutional-network-AU-tempuratue-prediction.py](1d-convolutional-network-AU-tempuratue-prediction.py) (tensorflow v1)

   1d-convolutional-network-AU-tempuratue-prediction.py includes a 1-d convolutional neural network forecasting model, built with TensorFlow v1.2. The model forecasts Australian    temperature using historical temperature/rainfall. 

## Particle Swarm Optimization on a GPU, Support Vector Machine 

#### support_vector_machine_gpu_optimization.ipynb (cuda, kernelml)

This notebook shows the optimization of a multi-class, linear support vector machine using a simulation based optimizer. Any simulation based optimizer could be used with the cuda kernel in this notebook. I used KernelML, my custom optimizer, in this example. 

Colab Link: https://colab.research.google.com/drive/1AptayjRoDITNLmyfCc0T7z_xKFBlg2l-#scrollTo=pa88P5JUvv_X

Blog Post:https://freedomtowin.github.io/2019/12/11/KernelML-SVM-GPU.html

The runtime for this script should be set to use the GPU: Runtime->Change runtime type.
