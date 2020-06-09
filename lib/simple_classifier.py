from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np

class Classification:
    def __init__(self, epoch = 1, batch_size = 5):
        self.epoch = epoch
        self.batch_size = batch_size
        self.es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience=10, baseline=0.20) # verbose=1 로 지정시, 언제 training을 멈췄는지 확인 가능, patience 성능이 증가하지 않는 것을 버티는 횟수, base line 특정 값에 도달 시 중지
        self.mc = ModelCheckpoint('best_____model.h5',monitor='val_loss',mode='min',save_best_only=True)

    def definition_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(8, activation='relu', input_shape=(2,)))
        self.model.add(layers.Dense(16, activation='relu'))
        # self.model.add(layers.Dense(32, activation='relu'))
        # self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(8, activation='softmax'))

    def compile_model(self):
        self.model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    def fit_model(self, train_x, train_y):
        self.model.fit(train_x, train_y, epochs=self.epoch, batch_size= self.batch_size, verbose = 1,callbacks=[self.es,self.mc])

    def accuracy_model(self, test_x, test_y):
        test_loss, test_acc = self.model.evaluate(test_x, test_y)
        print('test_acc: ', test_acc)

    def predict(self, data):
        prediction = self.model.predict(data)
        # print("original = ", np.argmax(test_labels[i]), "prediction = ", np.argmax(prediction[i]))
        return prediction