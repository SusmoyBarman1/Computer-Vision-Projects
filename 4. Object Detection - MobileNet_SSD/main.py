import cv2

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


def detectObjectFromImage(img, thress, model):
    classId, confidence, boundingBox = model.detect(img, confThreshold=thress)
    print(classId, boundingBox)

    cv2.imshow('Output', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # Paths
    imgPath = 'data/office.jpg'
    classFile = 'Files/coco.names'
    architecturePath = 'Files/architecture_ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'Files/weights_frozen_inference_graph.pb'
    cocoFile = 'Files/coco.names'

    cocoNames = []
    thres = 0.45
    
    cocoNames = getCOCOnames(classFile)
    #printClassNames(cocoNames)

    model = defineModel(architecturePath, weightsPath)

    img = cv2.imread(imgPath)
    detectObjectFromImage(img, thres, model)
