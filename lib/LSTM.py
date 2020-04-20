#<20.04.20> by KH
'''
capstone design project.

One of the modules in NILM

It should be tested later
'''
import numpy as np
import tensorflow as tf
import keras

class LSTM:
    def __init__(self):
        self.batch_size = 50
        self.input_size = 24
        self.output_size = 1

    def get_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def set_model(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv1D(filters=16, kernel_size=4, input_shape=(self.input_size,1) ,strides=1, padding='same', activation='linear'))    #input shape = (batch, steps, channels)
        self.model.add(keras.layers.LSTM(128))
        self.model.add(keras.layers.LSTM(256))
        self.model.add(keras.layers.Dense(128))
        self.model.add(keras.layers.Dense(self.output_size))

    def compile_fit_model(self):
        self.model.compile(loss='mean_squred_error', optimizer='adam', metrics = ['accuracy'])
        self.model.fit(self.x_train,self.y_train, batch_size=50, epochs=100, validation_data = (x_test, y_test))

    def print_model_summary(self):
        print(self.model.summaty())

    def operation(self,x_train,y_train, x_test, y_test):
        self.get_train_data(x_train, y_train)
        self.get_test_data(x_test, y_test)
        self.set_model()
        self.print_model_summary()
        self.compile_fit_model()

if __name__=="__main__":
    print("capstone design project.\n\
One of the modules in NILM\n\
It should be tested later")