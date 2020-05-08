'''
scaleFactor --  determines the factor by which the detection window of the classifier is scaled down per detection
                pass. A factor of 1.1 corresponds to an increase of 10%. Hence, increasing
                the scale factor increases performance, as the number of detection passes is reduced.
                However, as a consequence the reliability by which a face is detected is reduced.
minNeighbour=5 --  it is the minimum number of features which are if detected then it is considered that
                    the perticular area is a face & thus a face is detected , i.e. here, if  minimum 5 features are 
                    detected around an area then it is considered that it is a face . '''


import cv2
import matplotlib.pyplot as plt
import numpy as np
import model_set

face_cascades= cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
model_set.recognizer.read("trainner.yml")

model_set.set_of_people = {v:k for k,v in model_set.set_of_people.items()}
print(model_set.set_of_people)

def detect_face(img):
    Img = img.copy()
    found = face_cascades.detectMultiScale(Img,scaleFactor=1.2,minNeighbors=3)
    print(found,type(found))
    
    if found is () : # warning arises when a numpy array (here 'found') is compaired to python style empty lists/tuples.{see bookmarks} 
        cv2.putText(Img,'No face detected!',(50,50),cv2.FONT_ITALIC,2,(0,0,255),5,cv2.LINE_8)
        print("no face to detect")
    
    for (x,y,w,z) in found:
        cv2.rectangle(Img,(x,y),(x+w,y+z),(255,255,255),5)

        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# since LBPHFace recognizer only works on grey scale images
        roi = grey[y:y+z , x:x+w]

        id_ ,confidence = model_set.recognizer.predict(roi)
        print("id: ",id_ , ", confidence :", confidence)
        cv2.putText(Img,str(model_set.set_of_people[id_]),(x,y),cv2.FONT_ITALIC,2,(255,0,0),5,cv2.LINE_8)
        
    return Img


capture =cv2.VideoCapture(0)
while True:
    ret , frame=capture.read(0)
    frame = detect_face(frame)
    cv2.imshow("Face_detection",frame)
    if cv2.waitKey(1)==27:
        break

capture.release()
cv2.destroyAllWindows() 