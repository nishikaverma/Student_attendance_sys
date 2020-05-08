# validation and intermedite code
import os
import numpy as np
import cv2
from PIL import Image
import pickle


class controller:
    def __init__(self):
        
        self.trining_data = [] # 'ALL' the refrence images that the model will refer for recognizing faces i.e the training data set
        self.people_data = [] # the LABLES

        self.set_of_people = {} # the set of all the people (along with a unique id "count" given to each) whose images are being recognized
        self.face_cascades= cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        
        # self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def Image_paths(self):
        self.count=0 # an id for every person
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        req_path = os.path.join(BASE_DIR,"Images")

        for root,dirs,files in os.walk(req_path):
            for f in files:
                if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png") or f.endswith("PNG"):
                    # name of folder(on the name of person) of whose the image is:--
                    people_name = os.path.basename(root)
                    if people_name not in self.set_of_people: 
                        self.set_of_people[people_name]=self.count # asigning an id to every person(people_name) in set_of_people 
                        self.count+=1
                    id_ = self.set_of_people[people_name]

                    # path of image(each image that the model will use for refrence) :--
                    image_path = os.path.join(root,f) 

                    # converting each image into numpy array as 'detectMultiScale' only takes images as numpy array for  feature detevtion :--
                    image_array = np.array(Image.open(image_path).convert("L"))

                    found_face = self.face_cascades.detectMultiScale(image_array,scaleFactor=1.2,minNeighbors=3) #HERE, found_face == numpy array
                    for (x,y,w,z) in found_face:
                        roi_face = image_array[y:y+z , x:x+w]
                        self.trining_data.append(roi_face)
                        self.people_data.append(id_)

        # writting the set_of_people dict as a file:--                     
        my_file = open("Set_of_people_MYFILE","wb")
        pickle.dump(self.set_of_people,my_file)
        print("file written!")


    def Video_capture(self):
        
        self.capture =cv2.VideoCapture(0)
        while True:
            ret , frame = self.capture.read(0)
            frame = self.Face_detector(frame)
            cv2.imshow("Face_detection",frame)
            if cv2.waitKey(1)==27: # i.e. id Esc key is pressed
                break
        self.Video_distroy()
        

    def Video_distroy(self):
        self.capture.release()
        cv2.destroyAllWindows()       
        print("Window distroyed!")

    def Face_detector(self,img):
        self.isFound = False
        self.isUniuqe = True

        Img = self.img.copy()
        found = self.face_cascades.detectMultiScale(Img,scaleFactor=1.2,minNeighbors=3)
        print(found,type(found))
        if found is (): # if  no face detected
            pass
        elif (x,y,z,w) in found:
            pass

    def Tracking(self):
        pass 

    def Face_recognizer(self, parameter_list):
        pass
