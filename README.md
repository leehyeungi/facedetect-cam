# facedetect-cam
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

if not cap.isOpened():
    print("cap open failed")
    exit()

face_xml = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, img = cap.read()
    if not ret:
        print("Can't read cap")
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_xml.detectMultiScale(img_gray, 1.3, 5) #얼굴인식

    if len(faces) >= 2:
        
        x1,y1,w1,h1 = faces[0]
        x2,y2,w2,h2 = faces[1]
    
        mosaic_loc1 = img[y1:y1+h1, x1:x1+w1] #얼굴 부분 자르기
        
        #0번째 1번째 이미지 변경
        mosaic_loc = cv2.resize(mosaic_loc1, (x2+w2, y2+h2), cv2.INTER_LINEAR) #0 -> 1
        mosaic_loc = cv2.resize(mosaic_loc2, (x1+w1, y1+h1), cv2.INTER_LINEAR) #1 -> 0
        img_w_mosaic = img
        img_w_mosaic[y1:y1+h1, x1:x1+w1] =  mosaic_loc#원래 이미지 붙이기

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
