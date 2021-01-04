import cv2 
import argparse 
import os
  
# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
# capture frames from a camera
cap = cv2.VideoCapture('video.mp4')

count = 0
 
while cap.isOpened():            
    # reads frames from a camera
    ret, img = cap.read() 

    if not ret:
        print("no frame:(")
        break
 
    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    count = count + 1
  
    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # To draw a circle in a face 
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)

        #fps = cap.get(cv2.CAP_PROP_FPS)

        fpsPrint = 'Frame Count: ' + str(count)
        cv2.putText(img, fpsPrint, (x+10, y+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
  
  
    # Display an image in a window
    cv2.imshow('opencv face detection', img)
 
    # Wait for Esc key to stop    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
  
# Close the window
cap.release()
  
# De-allocate any associated memory usage
cv2.destroyAllWindows()