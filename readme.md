# REAL TIME FACIAL EXRESSION RECOGNITION
**Deep Learning + Machine Learning Engineering** project.

## INTRODUCTION
This is an application that detects facial expressions in real-time using a camera. The dataset used for model training is [FER-2013](https://www.kaggle.com/msambare/fer2013).
This app can detect 7 emotions - *angry, disgust, fear, happy, neutral, sad and surprise*. 

I built my own sequential convolution neural network with 4 convolution layers and 2 fully connected layers with the activation function - *Relu*.
The final layer consists of activation function - *Softmax* for final prediction.

## HOW TO RUN
- Use it as a Web Application on streamlit
1. Click on the link\
https://share.streamlit.io/shashankbhatia15/real-time-face-emotion-detector/main
2. Select device
3. Click on the start button. (it will take 2 mins to load after this step)
4. Give the required permission(s)
5. Click on stop when you are done.

- Run it on your local machine
1. install python
2. open cmd and run the following commands

   `git clone https://github.com/shashankbhatia15/Real-Time-Face-Emotion-detector.git`
   
   `cd Real-Time-Face-Emotion-detector`
   
   `pip install -r requirements.txt`
   
   `python camera_app.py`
   
  >NOTE - Press 'q' to quit

## Dependencies

Python\
Tensorflow\
Streamlit\
Streamlit-Webrtc\
OpenCV

## CNN Model


Convolutional Neural Network is one of the techniques to do image classification and image recognition in neural networks. A simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer.

![This is an image](https://static.javatpoint.com/tutorial/tensorflow/images/convolutional-neural-network-in-tensorflow.png)

**Architecture**:

1. INPUT layer will hold the raw pixel values of the image
2. CONVOLUTION layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume.
3. RELU layer will apply an elementwise activation function
4. POOL layer will perform a downsampling operation along the spatial dimensions (width, height), 
5. FC (i.e. fully connected) layer will compute the class scores and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.

## CONCLUSION
We have built a web application on streamlit using a CNN model with a validation accuracy of 67.3%. 







