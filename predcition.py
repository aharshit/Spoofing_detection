import numpy as np
import tensorflow as tf
import cv2
# Load the model from an HDF5 file
def get_padding_bbox_indices(x1, y1, w1, h1, real_w, real_h, ratio_bbox_and_image):
    x1_padding = x1 - int((w1) * (1 + ratio_bbox_and_image))
    y1_padding = y1 - int((h1) * (1 + ratio_bbox_and_image))
    w1_padding = w1 + int((w1) * (1 + ratio_bbox_and_image))
    h1_padding = h1 + int((h1) * (1 + ratio_bbox_and_image))
    if x1_padding < 0:
        x1_padding = 0
    if y1_padding < 0:
        y1_padding = 0
    if w1_padding > real_w:
        w1_padding = real_w
    if h1_padding > real_h:
        h1_padding = real_h
    return x1_padding, y1_padding, w1_padding, h1_padding

model = tf.keras.models.load_model('my_model.h5')
dim = (32, 32)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def preprocess(img,x1,y1,w,h):
    real_w = img.shape[1]
    real_h = img.shape[0]
    area_image = real_h * real_w
    area_bbox = w*h
    ratio_bbox_and_image=area_bbox / area_image
    x1_padding, y1_padding, w1_padding, h1_padding = get_padding_bbox_indices(x1, y1, w, h, real_w, real_h,ratio_bbox_and_image)
    padding_img = img[y1_padding:y1+h1_padding, x1_padding:x1+w1_padding]
    resized_padding_img = cv2.resize(padding_img, dim, interpolation = cv2.INTER_AREA)
    arr=np.asarray(resized_padding_img)
    return arr



# Convert the image to grayscale


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def predict_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            temp = preprocess(frame,x,y,w,h)
            img = np.reshape(temp, (1, 32, 32, 3))
            predicted_probabilities = model.predict(img)
            probabilities = softmax(predicted_probabilities)
            ans=-1
            for a, b in probabilities:
                if a > b:
                    color = (0, 0, 255)
                    ans=f"Spoof {a*100}"
                else:
                    color = (0, 255, 0)
                    ans = f"Live {b*100}"
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color,thickness)
            cv2.putText(frame, ans, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,color, 2)

        # Display the color frame with bounding boxes
        cv2.imshow('Face Detection', frame)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


predict_image()
