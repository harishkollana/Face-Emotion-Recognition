
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
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # load model
    model = load_model("Final_model_Custome_CNN.h5")
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']  # Emotion that will be predicted
    
except Exception:
    st.write("Error loading cascade classifiers")
    
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        
        img = frame.to_ndarray(format="bgr24")
        
        face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        
        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            image=gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img
        
        

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
