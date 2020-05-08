import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_tracker(): 
    # Create Tracker object:-
    tracker = cv2.TrackerMedianFlow_create()
    return tracker
   
def face_detection(frame,face_front_cascade): 
    found = face_front_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=3)
    
    if found is () : # warning arises when a numpy array (here 'found') is compaired to python style empty lists/tuples.{see bookmarks} 
        found=[]
        #face_detection(frame,face_front_cascade)
    return found

#CAPTURING 1st FRAME
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# Create MultiTracker object:-
multiTracker = cv2.MultiTracker_create()

#creating individual Tracker object for each face, detected on 1st(initial) frame through Multitrackier object 
face_front_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
init_faces = face_detection(frame_gray,face_front_cascade)

for (x, y, w, h) in init_faces:
    colour = (0,0,255)
    stroke = 20
    cv2.rectangle(frame,(x,y),(x+w,y+h), colour, stroke)
    bbox = (x,y,w,h)
    T = multiTracker.add(create_tracker(), frame, bbox)
    print( "Individual trackers cteated !")

while True:
    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
    
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    onframe_faces = face_detection(frame_gray,face_front_cascade)
    print("faces detected initially :", init_faces)
    print("faces detected onframe: ", onframe_faces,"\n---------\n")
    
    
    #######################################
    # When a new face enters on the frame :--
    if len(onframe_faces)>len(init_faces):
        print("@@@@@@@@@@@@@@@@@@@@@@@ INSIDE LOOP @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        for f in onframe_faces:
            
            if f not in init_faces:
                print("face to be added :", f) 
                print("initial_faces are (befor new face addition) :", init_faces)
                init_faces=np.append(init_faces,[f],axis=0)
                print("new init_faces list: ",init_faces)
                
                for (x, y, w, h) in [f]: 
                    #colour = (0,0,255)
                    #stroke = 20
                    #cv2.rectangle(frame,(x,y),(x+w,y+h), colour, stroke)
                    bbox = (x,y,w,h)
                    multiTracker.add(create_tracker(), frame, bbox)
                    print("tracker created")
                success, boxes = multiTracker.update(frame)
                print("tracker updated")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    '''
    ###################################
    # When a face leaves the frame :--
    if len(onframe_faces)<len(init_faces):
        print("######################## INSIDE LOOP #####################################")
        for f in init_faces:
            if f not in onframe_faces:
                print("face to be deleted :", f) 
                print("initial_faces are (befor face deletion) :", init_faces)
                init_faces=np.delete(init_faces,len(init_faces)-1,axis=0) # deleting face from init_faces
                print("new init_faces list: ",init_faces)
                
                # deleting the tracking object of that face
                for (x, y, w, h) in [f]: 
                    bbox = (x,y,w,h)
                    #multiTracker.add(create_tracker(), frame, bbox)
                    print("tracker deleted")
            success, boxes = multiTracker.update(frame)
            print("tracker updated")
    
    '''

    ###################################
    
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


cap.relese()
cv2.distroyAllWindows()