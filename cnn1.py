import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import imutils
import time

# Define the directories where the image datasets are located
wave_dir = "./datasets/waves/"
not_wave_dir = "./datasets/not_waves/"

# Load the image datasets
wave_images = [os.path.join(wave_dir, img) for img in os.listdir(wave_dir)]
not_wave_images = [os.path.join(not_wave_dir, img) for img in os.listdir(not_wave_dir)]

# Combine the datasets and create labels
images = wave_images + not_wave_images
labels = [1] * len(wave_images) + [0] * len(not_wave_images)

# Shuffle the datasets and split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
y_train = np.array(y_train)
y_test = np.array(y_test)


def preprocess_image(img_path):
    # Read the image
    img = cv2.imread(img_path)
    # Resize the image to (64, 64)
    img = cv2.resize(img, (64, 64))
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reshape the image to (64, 64, 1)
    img = np.reshape(img, (64, 64, 1))
    # Convert the pixel values to floats between 0 and 1
    img = img.astype('float32') / 255.0
    return img


X_train = np.array([preprocess_image(img_path) for img_path in X_train])
X_test = np.array([preprocess_image(img_path) for img_path in X_test])

model = load_model('wave_detection_model.h5')

# Create a VideoCapture object
cap = cv2.VideoCapture("wave_video.mp4")

# Define the parameters for the optical flow
prev_frame = None
hsv_mask = np.zeros((64, 64, 3), dtype=np.uint8)
hsv_mask[..., 1] = 255
flow_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=5,
    iterations=3,
    poly_n=5,
    poly_sigma=1.1,
    flags=0
)

# Define variables for tracking wave runup height
prev_height = None
wave_height = None

# Define the output video codec and parameters
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
fps = 30
out = cv2.VideoWriter("output.avi", fourcc, fps, (800, 600))

# Loop through each frame
while True:
    # Read the frame
    ret, frame = cap.read()

    # If the frame cannot be read, break the loop
    if not ret:
        break

    # Resize the frame and convert to grayscale
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow
    if prev_frame is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, **flow_params)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        ang = cv2.resize(ang, (64, 64))
        mag = cv2.resize(mag, (64, 64))
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr_mask = cv2.resize(cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR), (frame.shape[1], frame.shape[0]))

    # Detect waves in the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = np.reshape(img, (1, 64, 64, 1))
    img = img.astype('float32') / 255.0
    pred = model.predict(img)

    # Add bounding boxes to the detected waves
    if pred > 0.5:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 150)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Update the wave runup height
            if prev_height is not None:
                height_diff = y - prev_height
                if height_diff > 0:
                    wave_height = height_diff
            prev_height = y

    # Display the optical flow and annotations on the frame
    if prev_frame is not None:
        frame = cv2.add(frame, bgr_mask)
        cv2.putText(frame, "Wave Motion Run-up to the Left", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the wave runup height
        if wave_height is not None:
            cv2.putText(frame, "Wave Height: {:.2f} m".format(wave_height/600), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Write the frame to the output video
    out.write(frame)

    # Update the previous frame
    prev_frame = gray.copy()

    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object, release the output video, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
