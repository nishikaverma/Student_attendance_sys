import pickle

mydict={"apple":"Jobs" , "MS":"Gates" , "FB":"Zakurburg" ,"amazon":"Bezos" }

# to write it
f=open("company_founder","wb")
pickle.dump(mydict,f)
f.close()

print("files pickled!!")

#reading file
f=open("company_founder","rb")
my_file=pickle.load(f)
f.close()

print("file opened")