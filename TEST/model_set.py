import os 
import cv2
from PIL import Image
import numpy as np
import pickle

trining_data = [] # 'ALL' the refrence images that the model will refer for recognizing faces
people_data = [] # the LABLES

set_of_people = {} # the set of all the people (along with a unique id "count" given to each) whose images are being recognized
count=0 # an id for every person

face_cascades= cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
req_path = os.path.join(BASE_DIR,"Images")

for root,dirs,files in os.walk(req_path):
    for f in files:
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png"):
            # name of folder(on the name of person) of whose the image is:--
            people_name = os.path.basename(root)
            
            if people_name not in set_of_people:
                set_of_people[people_name]=count # asigning an id to every person(people_name) in set_of_people 
                count+=1
            id_ = set_of_people[people_name]
            
            # path of image(each image that the model will use for refrence) :--
            image_path = os.path.join(root,f) 
            
            # converting each image into numpy array as 'detectMultiScale' only takes images as numpy array for  feature detevtion :--
            image_array = np.array(Image.open(image_path).convert("L")) 

            found_face = face_cascades.detectMultiScale(image_array,scaleFactor=1.2,minNeighbors=3) #HERE, found_face == numpy array
            for (x,y,w,z) in found_face:
                roi_face = image_array[y:y+z , x:x+w]
                trining_data.append(roi_face)
                people_data.append(id_)

# writting the set_of_people dict as a file:--                     
my_file = open("Set_of_people_MYFILE","wb")
pickle.dump(set_of_people,my_file)
print("file written!")


recognizer.train(trining_data, np.array(people_data)) 
recognizer.save("D:/django_examples/STUDENT_ATTENDANCE_SYSTEM/TEST/trainner.yml")   
print("Trainner Created!")
print(set_of_people)
