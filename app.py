import numpy as np
import cv2
import streamlit as st
import keras
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classifier = keras.models.load_model('Final_model_Custome_CNN.h5')

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']  # Emotion that will be predicted

except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoProcessorBase):
    def transform(self, frame):
      label = []
      img = frame.to_ndarray(format="bgr24")
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


      #image gray
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(
          image=img_gray, scaleFactor=1.3, minNeighbors=1)

      for (x, y, w, h) in faces:
          a=cv2.rectangle(img=img, pt1=(x, y), pt2=(
              x + w, y + h), color=(255, 0, 0), thickness=2)
          roi_gray = img_gray[y:y + h, x:x + w]
          roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
          roi = roi_gray.astype('float')/255.0
          roi = img_to_array(roi)
          roi = np.expand_dims(roi,axis=0) ## reshaping the cropped face image for prediction
          prediction = classifier.predict(roi)[0]   #Prediction
          label=emotion_labels[prediction.argmax()]
          label_position = (x,y)
          b=cv2.putText(a,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
          label_position = (x, y)

      return b

def main():
    # Face Analysis Application #
    st.markdown("<h1 style='text-align:center'>Real Time Face Emotion Detection Application</h1>", unsafe_allow_html=True)
    
    #add space
    st.markdown("<br>", unsafe_allow_html=True)

    #add paragraph text and position it in the center
    st.markdown("<p style='text-align:center'>Click on start to use webcam and detect your face emotion</p>", unsafe_allow_html=True)
    
    #add space
    st.markdown("<br>", unsafe_allow_html=True)

    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
    #add space
    st.markdown("<br>", unsafe_allow_html=True)

    #add space
    st.markdown("<br>", unsafe_allow_html=True)
    
    html_temp4 = """
    <div style="background-color:#98AFC7;padding:10px">
      <h4 style="color:white;text-align:center;">This Application is developed by Harish using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose.</h4>
      <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
    </div>
    <br></br>
    <br></br>"""                       

    st.markdown(html_temp4, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
