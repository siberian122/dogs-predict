from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np

classes=['pomeranian','shiba','Yorkshireterrir']
num_classes=len(classes)
image_size=50

#main関数を定義
def main():
    X_train, X_test,y_train,y_test=np.load('./animal_aug.npy',allow_pickle=True)
    X_train=X_train.astype('float')/256
    X_test=X_test.astype('float')/256
    y_train=np_utils.to_categorical(y_train,num_classes)
    y_test=np_utils.to_categorical(y_test,num_classes)
    
    model=model_train(X_train,y_train)
    model_eval(model, X_test,y_test)
    
def model_train(X,y):
    #モデル層を積み重ねる形式、addで層を追加
    model=Sequential()
    
    model.add(Conv2D(32,(3,3),padding='same',input_shape=X.shape[1:]))
    #2次元畳み込み層
    #input_shape=(28,28,1) 28*28のグレースケール(白黒画像)を入力
    #Conv2D➞(3,3)などの中心のあるフィルタを32枚使用
    model.add(Activation('relu'))
    #relu Rectified Linear Unit(ランプ関数): 活性化関数0以下なら0、0以上ならそのまま
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #MaxPooling2D:2*2の最大プーリング層　入力内の2＊2の領域で最大の数値を出力
    model.add(Dropout(0.5))
    #過学習予防全結合の層とのつながりを25%無効化
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.75))
    #数字を上げるとlossが減る
    model.add(Flatten())
    #次元削減、1次元ベクトルに変換
    model.add(Dense(512))
    #ニューラルネットワークを追加、次元数は512
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    
    opt=keras.optimizers.RMSprop(lr=0.0001,decay=1e-6)
    #lr:学習率　decray:各更新の学習率減衰
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #loss…損失関数
    #optimezer…最適化アルゴリズムの指定
    
    model.fit(X,y,batch_size=32,epochs=75)
    
    model.save('./animal_dog_aug_cnn.h5')
    
    return model

def model_eval(model,x,y):
    scores=model.evaluate(x,y,verbose=1)
    print('Test Loss: ',scores[0])
    print('Test Accuracy: ',scores[1])
    
if __name__=='__main__':
    main()    
