import requests
import cv2
import numpy as np


url = 'http://192.168.0.102:8080/shot.jpg'

while True:
	img_receive = requests.get(url)
	img_arr = np.array(bytearray(img_receive.content), dtype=np.uint8)

	img = cv2.imdecode(img_arr, -1)

	cv2.imshow('AndroidCam', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
         break