import cv2
import face_recognition as fr

def detectFace(img):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fr.face_locations(img_rgb)

    count = 0
    
    for res in results:
        
        y1, x2, y2, x1 = res
        print(res)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

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