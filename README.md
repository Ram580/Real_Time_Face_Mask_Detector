# Real time Face Mask Detection App 

This repository implements a user-friendly application for real-time face mask detection using a mobilenet deep learning model.

## Features

- **Real-time Mask Detection:** Utilizes a pre-trained deep learning model to detect faces in real-time video streams and predict whether a person is wearing a mask or not.
- **Streamlit Integration:** The application is built using Streamlit, providing a user-friendly interface for controlling the video stream and displaying the mask detection results.
- **Visual Feedback:** Displays bounding boxes around detected faces and overlays labels indicating the predicted mask status (mask/no mask) along with confidence probabilities.
- **Efficient Model:** The model architecture employs MobileNetV2, known for its efficiency in real-time applications, striking a balance between performance and resource consumption.

## Technical Details

### Model Training

- **Dataset:** The model is trained using images of people wearing and not wearing masks, categorized into "with_mask" and "without_mask" classes.
- **Data Augmentation:** Data augmentation techniques such as rotation, zoom, and horizontal flip are applied to increase the diversity of training data.
- **Model Architecture:** MobileNetV2 is used as the base model for feature extraction, followed by additional layers for classification.
- **Training:** The model is trained using the Adam optimizer with a binary cross-entropy loss function.

### Mask Detection App

- **Face Detection:** Utilizes a pre-trained face detection model based on the Single Shot Multibox Detector (SSD) architecture to detect faces in the input video frames.
- **Model Loading:** The trained face mask detection model is loaded for inference during the real-time application.
- **Prediction:** For each detected face, the model predicts the presence of a mask, and the result is displayed on the video stream.
- **User Interface:** The application provides a simple interface for starting and stopping the video stream and displays real-time mask detection results.

## Why MobileNet?

MobileNetV2 is chosen for its efficiency and performance in real-time applications. It offers a smaller model size and requires fewer computations compared to traditional CNN architectures, making it ideal for deployment on resource-constrained devices or in scenarios where real-time performance is crucial.

### MobileNet and Efficiency:

MobileNetV2 is a popular choice for real-time applications due to its emphasis on efficiency. Compared to traditional CNNs, MobileNetV2 boasts a smaller size and requires fewer computations. This translates to faster processing of video frames, making it ideal for real-world scenarios where real-time performance is crucial. Despite its efficiency, MobileNetV2 maintains good accuracy on image classification tasks, striking a perfect balance between performance and resource consumption.

## Further Enhancements
- **Experiment with different pre-trained models for face detection and mask prediction to improve accuracy.
- **Train custom deep learning models for specific requirements or datasets.
- **Implement additional features such as counting the number of people with/without masks or integrating audio alerts for mask violations.
