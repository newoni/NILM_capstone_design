#<20.04.20> by KH
'''
capstone design project.

One of the modules in NILM

It should be tested later

LSTM input format : 3dim (n_observationsm, sequence_length, feater number)

'''
import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

class LSTM:
    def __init__(self):
        ####s
        # self.batch_size = 50
        self.batch_size = 1     # test data용 batch 1
        ####e
        self.input_size = 24
        self.output_size = 1
        self.epoch = 5          # check // compile_fit_model function 활용 시 유연하게 입력받을 수 있어야함.
        self.es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience=50, baseline=1) # verbose=1 로 지정시, 언제 training을 멈췄는지 확인 가능, patience 성능이 증가하지 않는 것을 버티는 횟수, base line 특정 값에 도달 시 중지
        self.mc = ModelCheckpoint('best_model.h5',monitor='val_loss',mode='min',save_best_only=True)

    def get_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def set_model(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv1D(filters=16, kernel_size=4, input_shape=(self.input_size,1) ,strides=1, padding='same', activation='linear'))    #input shape = (batch, steps, channels)
        self.model.add(keras.layers.LSTM(128,input_shape=(24,1),return_sequences = True))   # (time step ,feature)
        self.model.add(keras.layers.LSTM(256,input_shape=(128,1)))
        self.model.add(keras.layers.Dense(128))
        self.model.add(keras.layers.Dense(self.output_size))

    def compile_fit_model(self):
        self.model.compile(loss='MSE', optimizer='adam', metrics = ['accuracy'])
        self.model.fit(self.x_train,self.y_train, batch_size=self.batch_size, epochs=self.epoch, validation_data = (self.x_test, self.y_test),callbacks=[self.es,self.mc])

    def print_model_summary(self):
        print(self.model.summary())

    def operation(self,x_train,y_train, x_test, y_test):
        self.get_train_data(x_train, y_train)
        self.get_test_data(x_test, y_test)
        self.set_model()
        self.print_model_summary()
        self.compile_fit_model()