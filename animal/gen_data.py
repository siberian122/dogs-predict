from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes=['Chihuahua','shiba','Yorkshireterrir']
num_classes=len(classes)
image_size=50


x=[]
y=[]
for index, classlabel in enumerate(classes):
    photos_dir='./'+classlabel
    files=glob.glob(photos_dir+'/*.jpg')
    for i, file in enumerate(files):
        if i>=78: break
        image=Image.open(file)
        image=image.convert('RGB')
        image=image.resize((image_size,image_size))
        
        data=np.asarray(image)
        x.append(data)
        y.append(index)
        
x=np.array(x)
y=np.array(y)

X_train,X_test,Y_train,Y_test=model_selection.train_test_split(x,y)
xy=(X_train,X_test,Y_train,Y_test)
np.save('./animal.npy',xy)        