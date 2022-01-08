'''
Run this file to experiment with real-time facial emotion detection on your local machine with the help of a webcam.
Kindly refer the Readme file for instructions to run this code. 
'''

#import libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

# loading the model files
fc = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model =load_model(r'.model.h5')

#emotion list
emotions = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

#capture video from webcam
cap = cv2.VideoCapture(0)                                                       

#emotion prediction
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                               
    faces = fc.detectMultiScale(gray)                              

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),3)                     
        img_crop = gray[y:y+h,x:x+w]                                            
        img_crop = cv2.resize(img_crop,(48,48))    

        if faces is ():
            print("no faces detected")
            
        else:
            final_img = img_crop.astype('float')/255.0                              
            final_img = img_to_array(final_img)                                             
            final_img = np.expand_dims(final_img,axis=0)                                   

            prediction = model.predict(final_img)[0]                             
            label=emotions[prediction.argmax()]                           
            cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)  
                                
    cv2.imshow('Emotion Detector',frame)        

    #closing the app                                
    if cv2.waitKey(1) & 0xFF == ord('q') :                                       
        break

cap.release()
cv2.destroyAllWindows()
