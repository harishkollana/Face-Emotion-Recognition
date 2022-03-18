import cv2
import numpy as np
from tensorflow.keras.models import model_from_json  
from tensorflow.keras.preprocessing import image  
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

class Faceemotion(VideoTransformerBase):
  
  #load model  
  model = model_from_json(open("fer.json", "r").read())  

  #load weights  
  model.load_weights('fer.h5')  

  RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


  face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  


  cap=cv2.VideoCapture(0)  

  while True:  
      ret,test_img=cap.read()# captures frame and returns boolean value and captured image  
      if not ret:  
          continue  
      gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  

      faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  


      for (x,y,w,h) in faces_detected:
          print('WORKING')
          cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
          roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
          roi_gray=cv2.resize(roi_gray,(48,48))  
          img_pixels = image.img_to_array(roi_gray)  
          img_pixels = np.expand_dims(img_pixels, axis = 0)  
          img_pixels /= 255  

          predictions = model.predict(img_pixels)  

          #find max indexed array  
          max_index = np.argmax(predictions[0])  

          emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  
          predicted_emotion = emotions[max_index]  
          print(predicted_emotion)
          cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  

      resized_img = cv2.resize(test_img, (1000, 700))  
      b=cv2.imshow('Facial emotion analysis ',resized_img)  

  return b

def main():
  # Face Analysis Application #
  st.title("Live Class Monitoring System")
  
  html_temp = """
      <body style="background-color:red;">
      <div style="background-color:teal ;padding:10px">
      <h2 style="color:white;text-align:center;">Face Emotion Recognition WebApp</h2>
      </div>
      </body>
          """
  st.markdown(html_temp, unsafe_allow_html=True)
  st.header("Webcam Live Feed")
  st.write("Click on start to use webcam and detect your face emotion")
  webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                  video_processor_factory=Faceemotion)
        
        
  if __name__ == "__main__":
    main()
