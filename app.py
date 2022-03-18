# -*- coding: utf-8 -*-

import os
import opencv-python
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode


# Loading pre-trained parameters for the cascade classifier
try:
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # load model
    model = load_model("Final_model_Custom_CNN.h5")
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']  # Emotion that will be predicted
except Exception:
    st.write("Error loading cascade classifiers")
    
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        label=[]
        img = frame.to_ndarray(format="bgr24")
        face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        
        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return frame
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]

            # Resizing the image to 48x48 so that the CNN can be fed
            roi_gray = cv2.resize(roi_gray, (48,48), interpolation = cv2.INTER_AREA)
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0) # Add a new axis at the end of the array
            prediction = model.predict(roi)[0] # Predicting the emotion of the face
            label = emotion_labels[np.argmax(prediction)]
            label_position = (x,y)
            frame = cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return frame
        
        

def main():
    # Face Analysis Application #
    st.title("Live Class Monitoring System")
    st.subheader("Face Analysis")
    st.write("This application will detect faces in the video stream and predict the emotion of the person.")
    st.write("The prediction is based on the pre-trained model.")
    st.write("The model is trained on the dataset of Emotion dataset.")

    # WebRTC streamer
    st.write("Click on the button below to start the stream.")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

if __name__ == '__main__':
    main()
