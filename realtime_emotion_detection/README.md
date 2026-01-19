Real-Time Emotion Detection using CNN and OpenCV

Project Overview  
This project implements a real-time facial emotion recognition system using Convolutional Neural Networks and OpenCV.  
The system captures live video from a webcam, detects human faces, and classifies facial expressions into different emotion categories in real time.

The model is trained on the FER2013 facial emotion dataset and then deployed for live emotion detection.

--------------------------------------------------

Objectives  
The main objective of this project is to detect human emotions from facial expressions using deep learning techniques.  
The system performs face detection, emotion classification, and real-time visualization on webcam video.

--------------------------------------------------

Emotion Classes  
The system can recognize the following seven emotions  
Angry  
Disgust  
Fear  
Happy  
Sad  
Surprise  
Neutral  

--------------------------------------------------

Model Description  
A Convolutional Neural Network is used for emotion classification.  
The model consists of convolution layers for feature extraction, max pooling layers for dimensionality reduction, dropout layers to prevent overfitting, and fully connected dense layers.  
Softmax activation is used in the output layer to classify multiple emotion categories.

--------------------------------------------------

Technology Stack  
Programming Language: Python  
Deep Learning Framework: TensorFlow and Keras  
Computer Vision Library: OpenCV  
Numerical Library: NumPy  
Visualization Library: Matplotlib  

--------------------------------------------------

Project Structure  
realtimedetection folder contains  
train_emotion_model.py file for training the CNN model  
realtimeemotion.py file for real-time emotion detection using webcam  
emotion_model.h5 trained model file  
train folder containing training images for each emotion class  
test folder containing testing images for each emotion class  

--------------------------------------------------

Dataset Information  
The FER2013 dataset is used for training and testing.  
It contains more than thirty five thousand grayscale facial images of size 48 by 48 pixels.  
The dataset includes seven emotion classes.  
The dataset was downloaded from Kaggle.

--------------------------------------------------

How to Run the Project  

Step 1  
Install the required libraries using pip  
tensorflow  
opencv python  
numpy  
matplotlib  

Step 2  
Run the training file  
python train_emotion_model.py  
This will train the CNN model and save the trained model as emotion_model.h5

Step 3  
Run the real-time detection file  
python realtimeemotion.py  
The webcam will open and emotions will be detected in real time  
Press q to exit the application

--------------------------------------------------

Results  
The model achieved around fifty five percent validation accuracy on the FER2013 dataset.  
The system performs real-time emotion detection smoothly on CPU.

--------------------------------------------------

Applications  
Human computer interaction  
Emotion aware systems  
Student academic projects  
Computer vision learning projects  

--------------------------------------------------

Notes  
Good lighting conditions improve detection accuracy.  
Webcam quality can affect performance.  
Python version 3.10 is recommended for better TensorFlow compatibility.

--------------------------------------------------

Author  
Name: Avadhut Yashwant Mote  
Project Type: Academic Project  

--------------------------------------------------

License  
This project is intended for educational and learning purposes only.

--------------------------------------------------
