import numpy as np 
import os

print("this file is (__file__):",__file__)
print("abspath",os.path.abspath(__file__))
print("dirname:",os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path1= os.path.join(BASE_DIR,"TEST")
print(path1)

print("___________________________________________")

for root,dirs,files in os.walk(path1):
    print('basename: ',os.path.basename(root))
    print(root)
    print(dirs)
    print(files)
    print("\n")
    