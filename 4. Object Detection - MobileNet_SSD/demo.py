import numpy as np

data = np.array([[0.84345317],
                 [0.62138087],
                 [0.12347777]])

data = list(data.reshape(1, -1)[0])
data = list(map(float, data))

print(type(data), data)


'''for classId, conf, box in zip(classIds.flatten(), confidence.flatten(), boundingBox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, cocoNames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Detection Output', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break'''


#print(boundingBox, confidence)
        #print(type(boundingBox), type(confidence))
        #print(classIds.flatten())
        #print(confidence.flatten())


'''imgPath = 'data/tesla.jpg'
    img = cv2.imread(imgPath)'''