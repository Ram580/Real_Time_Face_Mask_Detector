import streamlit as st
import cv2
import detect_mask_video 
from tensorflow.keras.models import load_model
import numpy as np
import imutils
from imutils.video import VideoStream

# Import necessary libraries
import streamlit as st
import cv2
import numpy as np
from detect_mask_video import detect_and_predict_mask  # Import function from your module
import time

# Streamlit app
def main():
    st.title("Real-time Face Mask Detection")

    activities = ["Home", "Real-time Video"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        st.subheader("Real-time Face Mask Detection")

    elif choice == "Real-time Video":
        st.subheader("Real-time Video")
        st.text("Click Start to start the real-time video stream")

        # Create a button for starting/stopping the video stream
        start_button = st.button("Start Video Stream")

        # Create a placeholder to display the video frames
        frame_placeholder = st.empty()

        # Initialize video stream
        #vs = cv2.VideoCapture(0)
        
        # load our serialized face detector model from disk
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk 
        maskNet = load_model("mask_detector.h5") 
        # Load the SavedModel
        #maskNet = TFSMLayer("mask_detector.model", call_endpoint='serving_default')

        # initialize the video stream
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()

        if start_button:
            while True:
                # initialize the video stream
                #print("[INFO] starting video stream...")
                vs = VideoStream(src=0).start()
                # Capture frame-by-frame
                frame = vs.read()
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                #frame = vs.read()
                frame = imutils.resize(frame, width=400)

                # Detect faces in the frame and determine if they are wearing a mask or not
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                # Loop over the detected face locations and their corresponding locations
                for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                    # Code from your detect_mask_video.py file
                    # ...

                # Display the resulting frame
                frame_placeholder.image(frame, channels="BGR")
                
                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # If the 'Stop Video Stream' button is pressed, stop the video stream
                if st.button('Stop Video Stream'):
                    break

            # When everything done, release the capture
            vs.release()

if __name__ == "__main__":
    main()
