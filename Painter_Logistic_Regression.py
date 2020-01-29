#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split

def resize(img):
    new_width  = 200
    new_height = 200 
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    return img

def image_dir(direc):
    Image_directory=direc
    all_clips = set([f for f in listdir(Image_directory) if isfile(join(Image_directory, f))])
    imgs=[] 
    img_t=[]
    img_p=[]
    print('Images are loaded')
    for num, img in enumerate(all_clips):
        imgss=Image.open(Image_directory+img)
        imgss=resize(imgss)
        img_t.append(numpy.array(imgss))
    for j in range(0,len(img_t)):
        try:
            img_p.append(img_t[j].reshape(200,200,3))
        except:
            pass
        
    return img_p
        
picasso    = image_dir(direc='C:/Users/PRATHAMESH/Desktop/best-artworks-of-all-time/images/images/Pablo_Picasso/')
picasso_cut= picasso[:262]

rembrant   = image_dir(direc='C:/Users/PRATHAMESH/Desktop/best-artworks-of-all-time/images/images/Rembrandt/')

dali       = image_dir(direc='C:/Users/PRATHAMESH/Desktop/best-artworks-of-all-time/images/images/Salvador_Dali/')
        
final_array=picasso_cut+rembrant+dali
picasso_out=[0]
picasso_out=picasso_out*len(picasso_cut)
rembrant_out=[1]
rembrant_out=rembrant_out*len(rembrant)
dali_out=[3]
dali_out=dali_out*len(dali)

Y=picasso_out+rembrant_out+dali_out

X_train, X_test, Y_train, Y_test = train_test_split(final_array,Y, test_size=0.3, random_state=0)

def normalize(X):
    for i in range(0,len(X)):
        X[i]=X[i]/255
    return X
    
X_train = normalize(X_train)
X_test = normalize(X_test)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(200,200,3)),
    keras.layers.Dense(256, activation='relu'),
#     keras.layers.Flatten(input_shape=(200,200,3)),
#     keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(20, activation='softmax')
])

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

model.fit(numpy.asarray(X_train),numpy.array(Y_train), epochs=5)
test_loss, test_acc=model.evaluate(numpy.asarray(X_test), numpy.asarray(Y_test))
prediction=model.predict(numpy.asarray(X_test), max_queue_size=200)
i=list(prediction)
out_list=[]
for i in out:
    out_list.append(int(i))
    
for i in range(50):
    plt.grid(False)
    plt.imshow(X_test[i])
    plt.xlabel('Actual '+str(Y_test[i]))
    plt.ylabel('Prediction '+str(numpy.argmax(out[i])))
    plt.show()
    if Y_test[i]!=out[i]:
        plt.show()

