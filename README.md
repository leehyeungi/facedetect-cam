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
    for (x, y, w, h) in faces:
        mosaic_loc = img[y:y+h, x:x+w] #얼굴 부분 자르기
        
        mosaic_loc = cv2.blur(mosaic_loc,(50,50))
        img_w_mosaic = img
        img_w_mosaic[y:y+h, x:x+w] = mosaic_loc #원래 이미지 붙이기

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
