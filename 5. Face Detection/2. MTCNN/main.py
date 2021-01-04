import cv2
import mtcnn

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


face_detector = mtcnn.MTCNN()

def detectFace(img):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img_rgb)

    count = 0
    
    for res in results:
        count = count + 1
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
           
        cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)

    return img



if __name__ == '__main__':

    vc = cv2.VideoCapture('video.mp4')

    while vc.isOpened():
        ret, frame = vc.read()
        
        if not ret:
            print("no frame:(")
            break

        frame = detectFace(frame)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break