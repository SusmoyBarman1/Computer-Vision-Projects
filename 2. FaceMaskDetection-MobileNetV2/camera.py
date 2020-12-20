from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import mtcnn

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


model = './model/mask_no_mask.h5'
videoPath = './dataset/video/sus1.mp4'
model = load_model("./model/mask_no_mask.h5")

face_detector = mtcnn.MTCNN()

def recognize(img, model):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img_rgb)
    
    for res in results:
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        check = False
        name = 'No mask'
    
        if res['confidence'] > .9:
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.array(face, dtype="float32")
            face = np.expand_dims(face, axis=0)
            val = model.predict(face)[0]
            if val > .5:
                check = True
                name = 'Mask'
        
            #print(face.shape)
    
        
        if check:        
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    return img

vc = cv2.VideoCapture(videoPath)

while vc.isOpened():
    ret, frame = vc.read()
    print(ret)
    if not ret:
        print("no frame:(")
        break
    frame = recognize(frame, model)
    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break