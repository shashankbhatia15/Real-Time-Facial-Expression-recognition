'''
This is a script to run the project as a web application on streamlit.
simply click on the link provided in the Readme file to watch a demonstration.
'''

#import libraries
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

# function to configure webcam for streamlit and turn off mic
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# importing the model
faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model =load_model(r'model.h5')

#emotions
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


# emotion prediction 
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray)
        if faces is ():
            print("no faces detected")

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
            img_crop = gray[y:y+h,x:x+w]
            img_crop = cv2.resize(img_crop,(48,48))

            final_img = img_crop.astype('float')/255.0
            final_img = img_to_array(final_img)
            final_img = np.expand_dims(final_img,axis=0)

            prediction = model.predict(final_img)[0]
            label=emotion_labels[prediction.argmax()]
            cv2.putText(img,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)     
        return img

webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )
        


