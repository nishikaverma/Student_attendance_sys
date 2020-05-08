import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_tracker():
    # Create Tracker object:-
    tracker = cv2.TrackerMedianFlow_create()
    return tracker

#CAPTURING 1st FRAME
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
print("frame :" ,frame)
print(type(frame))

# selecting  MULTIPLE roi's from the frame:-
# FOR THIS :-@ Either use cv2.selectROI in a loop i.e. multiple times to take many roi's.
#                - cv2.selectROI returns a tuple i.e. (x,y,w,z)
#            @ OR use cv2.selectROIs that allows to take multiple roi's from a frame.
#                 - cv2.selectROIs returns a list of roi's as numpy arrays i.e. [ [x y w z ], [x y w z ],...] 
 
bbox = cv2.selectROIs("training",frame,False)
print("Roi's are : ",bbox)
print("TYpe of roi : ",type(bbox))

# Create MultiTracker object:-
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker 
for b in bbox:
    multiTracker.add(create_tracker(), frame, tuple(b))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)
    print("Success :", success)
    print(" Upadated ROI's :", boxes)

    # draw tracked objects
    if success:
        for i, box in enumerate(boxes):
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)    
    else:
        cv2.putText(frame,"Failed to track!",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    
    # show frame
    cv2.imshow('MultiTracker', frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

cap.relese()
cv2.distroyAllWindows()