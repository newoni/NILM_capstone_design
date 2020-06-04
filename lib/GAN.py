import numpy as np
from keras.models import Model, Sequential
from keras.layers import *

'''
input data: array. put for generate fake data 

**************************************************

NOTICE
generator's output shape should be same as discriminator's input shape

discriminator's output shape will be 1. cause I will make that my train_y 
'''
class GAN:

    def __init__(self, input_data, epochs, batch_size, batch):
        self.input_data = input_data
        self.shape= np.shape(input_data)

        self.epochs = epochs
        self.batch_size = batch_size
        self.batch = batch

        self.D = self.discriminator()
        self.G = self.generator()
        self.GD = self.combined()

    def discriminator(self):
        """
        define discriminator
        """
        D = Sequential()
        D.add(Conv1D(filters = 10, kernel_size= 3, padding='same', input_shape = (10,1)))
        D.add(LeakyReLU(0.2))
        D.add(Conv1D(filters=1, kernel_size=3, padding='same'))
        D.add(LeakyReLU(0.2))
        D.add(Reshape((-1,)))
        D.add(Dense(1, activation='sigmoid'))

        D.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return D

    def generator(self):
        """
        define generator
        """
        G = Sequential()
        G.add(Dense(30, input_shape=(10,)))     # (n, 30) 출력 수정 시 69번째 줄도 같이 수정
        G.add(LeakyReLU(0.2))
        G.add(Dense(128))                       # (n, 128 출력)
        G.add(Reshape((128, -1)))               # (, 128, 1 출력)
        G.add(Conv1D(filters =10, kernel_size =2, padding='same', activation='relu'))
        G.add(Conv1D(filters = 1, kernel_size =2, padding='same') )
        G.add(LeakyReLU(0.3))
        G.add(Reshape((-1,)))       #(n, 128) 출력
        G.add(Dense(10, activation='relu'))
        G.add(Reshape((10,1)))

        G.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return G

    def combined(self):
        """
        define combined gan model
        """
        G, D = self.G, self.D
        D.trainable = False
        generator_input = Input(shape=(10,))   # 수정 시 mk noise 함수 input 단 같이 수정
        generator_output = G(generator_input)
        discriminator_output = D(generator_output)
        GD = Model(input=generator_input, output= discriminator_output)

        GD.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        D.trainable = True
        return GD

    def mk_noise(self, shape=10): # check batch 사이즈 좀 데이터의 형태에 따라 좀 더 유연하게 짤 수 있도록 생각해보기 try except 사용해야할 듯.
        '''
        make noise one. that is input of generator
        :return:
        '''
        self.noise_data = np.random.rand(self.batch_size, shape)

    def mk_fake(self, noise_data):
        fake_data_x = self.G.predict(noise_data, batch_size = self.batch_size)
        fake_data_y = [0]*self.batch_size
        return fake_data_x, fake_data_y

    def mk_real(self):
        real_batch_x = self.input_data[np.random.randint(0,self.input_data.shape[0], size =self.batch_size)]
        real_batch_x = real_batch_x.reshape(np.shape(real_batch_x)[0],np.shape(real_batch_x)[1],1)
        real_batch_y = [1]*self.batch_size
        return real_batch_x, real_batch_y

    def mk_train_data_for_discriminator(self):
        self.mk_noise()
        fake_train_x, fake_train_y = self.mk_fake(self.noise_data)
        real_train_x, real_train_y = self.mk_real()
        train_x = np.concatenate( (fake_train_x,real_train_x ), axis= 0)
        train_y = fake_train_y + real_train_y
        return train_x, train_y

    def training_combined_model(self):
        train_y = [1]*self.batch_size
        self.D.trainable = False

        self.GD.train_on_batch(self.noise_data, train_y)

    def training_discriminator_model(self):
        train_x, train_y = self.mk_train_data_for_discriminator()
        self.D.trainable = True
        self.D.train_on_batch(train_x, train_y)

    def iter_oper(self):
        print("total epochs:", self.epochs)
        for i in range(self.epochs):
            print("epoch:", i)
            for j in range(self.batch):
                self.mk_noise()
                for k in range(5):
                    self.training_combined_model()
                self.training_discriminator_model()

    def cp_data(self):
        self.mk_noise()
        return self.G.predict(self.noise_data).reshape(-1,10)