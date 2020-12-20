import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('img.jpg')

#print(pytesseract.image_to_string(img))

################################################
'''
### Detect Characters


height, weight, _ = img.shape
boxes = pytesseract.image_to_boxes(img)

for b in boxes.splitlines():
    #print(b)
    char, x, y, w, h = b.split(' ')[:5]
    x, y, w, h = int(x), int(y), int(w), int(h)

    cv2.rectangle(img, (x, height-y), (w, height-h), (0, 0, 255), 2)
    cv2.putText(img, char, (x+8, height-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

'''
#####################################################

'''
### Detect words
'''    

height, weight, _ = img.shape
boxes = pytesseract.image_to_data(img)

for x, b in enumerate(boxes.splitlines()):
    
    if x!=0:
        data = b.split()
        print(data)
        
        if len(data)==12:
            x, y, w, h = int(data[6]), int(data[7]), int(data[8]), int(data[9])

            cv2.rectangle(img, (x, y), (w+x, h+y), (0, 0, 255), 2)
            cv2.putText(img, data[11], (x+8, y+100), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)


cv2.imshow('Result', img)
cv2.waitKey(0)