import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('C:/Users/Ganesh/Desktop/digicv/src/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade=cv2.CascadeClassifier('C:/Users/Ganesh/Desktop/digicv/src/cascades/data/haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('C:/Users/Ganesh/Desktop/digicv/src/cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

label={"person_name":1}
with open("label.pickle",'rb') as f:
    og_lables=pickle.load(f)
    labels={v:k for k,v in og_lables.items()}

cap = cv2.VideoCapture(0)

while(True):
   
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]

        id_, conf=recognizer.predict(roi_gray)
        if conf>=45:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255, 255, 255)
            stroke=2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            smile=smile_cascade.detectMultiScale(roi_gray)
            eyes=eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
              #  cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


        #img_item="7.png"
        #cv2.imwrite(img_item, roi_gray)

        color=(255,0,0)
        stroke=2
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)


    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
