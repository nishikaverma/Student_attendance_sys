1import cv2
import numpy as np
import matplotlib.pyplot as plt

def choose_tracker():
    
    print("enter 1 for TrackerBoosting_create function ")
    print("enter 2 for TrackerMIL_create function ")
    print("enter 3 for TrackerKCF_createfunction ")
    print("enter 4 for .TrackerTLD_create function ")
    print("enter 5 for .TrackerMedianFlow_create function ")
    choice = int(input("which tracker would you like to choose?"))
    if choice ==1:
        tracker=cv2.TrackerBoosting_create()
    if choice ==2:
        tracker=cv2.TrackerMIL_create()
    if choice ==3:
        tracker=cv2.TrackerKCF_create()
    if choice ==4:
        tracker=cv2.TrackerTLD_create()
    if choice ==5:
        tracker=cv2.TrackerMedianFlow_create()
    if choice ==5:
        tracker=cv2.Multi()
        
    return tracker
    
tracker=choose_tracker()
tracker_name=str(tracker).split()[0][1:]

print("You choose :-- ",tracker_name)

cap = cv2.VideoCapture(0)
ret,frame = cap.read()

print("frame :" ,frame)
print(type(frame))

#img=plt.imread("nihsi3.jpg")
#rects=[]
#roi = cv2.selectROI(img,False,False) # for detectinig a single object

bbox = cv2.selectROIs("training",frame,False)
print("Roi's are : ",bbox)
print("TYpe of roi : ",type(bbox))

bbox=tuple(bbox[0])

print(bbox)
ret = tracker.init(frame ,bbox)
print(ret)

while True:
    ret ,frame =cap.read()
    
    success, roi =tracker.update(frame)
    print("success :", success," ,  ROI :", roi)
    
    (x,y,w,z)=tuple(map(int,roi))
    
    if success:
        cv2.rectangle(frame,(x,y),(x+w,y+z),(0,255,0),3)
    else:
        cv2.putText(frame,"Failed to track!",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        
    cv2.imshow(tracker_name,frame)
        
    
    k= cv2.waitKey(1) & 0xFF
    if k==27:
        break
        
cap.relese()
cv2.distroyAllWindows()