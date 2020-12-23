from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes=['pomeranian','shiba','Yorkshireterrir']
num_classes=len(classes)
image_size=50
num_testdata=60


x_train=[]
x_test=[]
y_train=[]
y_test=[]

for index, classlabel in enumerate(classes):
    photos_dir='./'+classlabel
    files=glob.glob(photos_dir+'/*.jpg')
    for i, file in enumerate(files):
        if i>=71: break
        image=Image.open(file)
        image=image.convert('RGB')
        image=image.resize((image_size,image_size))
        
        data=np.asarray(image)
        if i< num_testdata:
            x_test.append(data)
            y_test.append(index)
        else:
            for angle in range(-20,20,2):
                #回転
                img_r=image.rotate(angle)
                data=np.asarray(img_r)
                x_train.append(data)
                y_train.append(index)
                
                #反転
                img_trans=image.transpose(Image.FLIP_LEFT_RIGHT)
                data=np.asarray(img_trans)
                x_train.append(data)
                y_train.append(index)
                
        
#x=np.array(x)
#y=np.array(y)
x_test=np.array(x_test)
x_train=np.array(x_train)
y_test=np.array(y_test)
y_train=np.array(y_train)

#X_train,X_test,Y_train,Y_test=model_selection.train_test_split(x,y)
xy=(x_train,x_test,y_train,y_test)
np.save('./animal_aug.npy',xy)        