import cv2
import numpy as np

def getCOCOnames(classFile):
    with open(classFile, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    
    return names

def printClassNames(classNames):
    for i, name in enumerate(classNames) :
        print(i+1, name)

def defineModel(architecturePath, weightsPath):
    net = cv2.dnn_DetectionModel(weightsPath, architecturePath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    return net


def detectObjectFromImage(thress, model, cocoNames):

    '''imgPath = 'data/tesla.jpg'
    img = cv2.imread(imgPath)'''

    videoPath = 'data/video.mp4'
    vc = cv2.VideoCapture(videoPath)

    while vc.isOpened():
        ret, img = vc.read()
        if not ret:
            print("no frame:(")
            break
        
        classIds, confidence, boundingBox = model.detect(img, confThreshold=thress)
        #print(classIds.flatten())
        #print(confidence.flatten())

        for classId, conf, box in zip(classIds.flatten(), confidence.flatten(), boundingBox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, cocoNames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Detection Output', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # Paths
    classFile = 'Files/coco.names'
    architecturePath = 'Files/architecture_ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'Files/weights_frozen_inference_graph.pb'
    cocoFile = 'Files/coco.names'

    cocoNames = []
    thres = 0.45
    
    cocoNames = getCOCOnames(classFile)
    #printClassNames(cocoNames)

    model = defineModel(architecturePath, weightsPath)

    detectObjectFromImage(thres, model, cocoNames)
