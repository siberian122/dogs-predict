from keras.models import Sequential, load_model
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras, sys
import numpy as np
from PIL import Image

classes=['pomeranian','shiba','Yorkshireterrir']
num_classes=len(classes)
image_size=50

def build_model():
    #モデル層を積み重ねる形式、addで層を追加
    model=Sequential()
    
    model.add(Conv2D(32,(3,3),padding='same',input_shape=(50,50,3)))
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
    #過学習予防全結合の層とのつながりを50%無効化
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    #数字を上げるとlossが減る(当たり前)
    model.add(Flatten())
    #次元削減、1次元ベクトルに変換
    model.add(Dense(512))
    #ニューラルネットワークを追加、次元数は512
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    
    opt=keras.optimizers.RMSprop(lr=0.0001,decay=1e-6)
    #lr:学習率　decray:各更新の学習率減衰
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #loss…損失関数
    #optimezer…最適化アルゴリズムの指定
    
    #モデルロード
    model=load_model('./animal_dog_aug_cnn.h5')
    
    return model

def main():
    image=Image.open(sys.argv[1])
    image=image.convert('RGB')
    image=image.resize((image_size, image_size))
    data=np.asarray(image)/255#粒度を下げる
    X=[]
    X.append(data)
    X=np.array(X)
    model=build_model()
    
    result=model.predict([X])[0]#結果返却
    predicted=result.argmax()
    percentage = int(result[predicted]*100)
    print('{}({}%)'.format(classes[predicted],percentage))
    
if __name__=='__main__':
    main()
    
    