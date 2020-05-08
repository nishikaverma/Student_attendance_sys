import cv2 as cv

cap = cv.VideoCapture(0)

face_front_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")    
tracker = cv.TrackerMedianFlow_create()
multiTracker = cv.MultiTracker_create()
bbox = ()

while True:
    ret,frame = cap.read()

    #press S to capture the face
    if cv.waitKey(20) & 0xFF == ord("s"):
        frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        face = face_front_cascade.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=3)
        print(face)
        for (x, y, w, h) in face: 
            colour = (0,0,255)
            stroke = 20
            cv.rectangle(frame,(x,y),(x+w,y+h), colour, stroke)
            bbox = (x,y,w,h)
            # creating seprate tracker for every face:-
            tracker = cv.TrackerKCF_create() #overwrite old tracker
            multiTracker.add(tracker, frame, bbox)



    #trace face and draw box around it
    print("boxes :", bbox)
    if bbox:
        #tracker.init(frame, bbox)   
        #ret, bbox = tracker.update(frame)
        ret, boxes = multiTracker.update(frame)
        print("From multitrack update, boxes : ",boxes)   
        
        if ret:
            for i, box in enumerate(boxes):
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv.rectangle(frame, p1, p2, (0,255,0), 2, 1)    
        else:
            cv.putText(frame,"Failed to track!",(100,200),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    
        
    #show result
    cv.imshow("frame",frame)

    #press ESC to exit
    if cv.waitKey(20) & 0xFF ==27:
        break    
cap.release()
cv.destroyAllWindows()

