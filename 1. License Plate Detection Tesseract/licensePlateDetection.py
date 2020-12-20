import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('banglaLicensePlate.jpg')
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

reverseImage = cv2.bitwise_not(blackAndWhiteImage)


'''
### Detect Characters

height, weight = blackAndWhiteImage.shape
boxes = pytesseract.image_to_boxes(blackAndWhiteImage)

for b in boxes.splitlines():
    print(b)
    char, x, y, w, h = b.split(' ')[:5]
    x, y, w, h = int(x), int(y), int(w), int(h)

    cv2.rectangle(blackAndWhiteImage, (x, height-y), (w, height-h), (0, 0, 255), 2)
    cv2.putText(blackAndWhiteImage, char, (x+8, height-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)


'''
#########################################
'''
### Detect Words

height, weight = reverseImage.shape
boxes = pytesseract.image_to_data(reverseImage)

for x, b in enumerate(boxes.splitlines()):
    
    if x!=0:
        data = b.split()
        print(data)
        
        if len(data)==12:
            x, y, w, h = int(data[6]), int(data[7]), int(data[8]), int(data[9])

            cv2.rectangle(reverseImage, (x, y), (w+x, h+y), (0, 0, 255), 6)
            #cv2.putText(img, char, (x+8, height-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

'''

cv2.imshow('Reverse', blackAndWhiteImage)
cv2.waitKey(0)