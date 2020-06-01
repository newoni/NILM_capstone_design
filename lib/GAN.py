import numpy as np
from keras.models import Sequential,Model
from keras.layers import *
from keras.datasets import mnist
from keras.optimizers import  Adam
from keras.layers.advanced_activations import LeakyReLU
from tqdm import tqdm

class GAN:
     (X_train, Y_train),(X_test, Y_test)=mnist.load_data()
     X_train = np.reshape(X_train, (60000,28,28,1)).astype('float32')
     X_test = np.reshape(X_test, (10000,28,28,1)).astype('float32')
     X_train = (X_train-127.5)/127.5

    # generator 모델 생성
    def generator_model():
        model = Sequential()
        model.add(Dense(128 * 7 * 7, input_dim=100, activation=LeakyReLU(0.1)))
        model.add(BatchNormalization())
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, (5, 5), padding='same', activation=LeakyReLU(0.1)))
        model.add(UpSampling2D())
        model.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))
        return model

    generator_model = generator_model()

    # discriminator 모델 생성
    def discrimnator_model():
        model = Sequential()
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1), activation=LeakyReLU(0.1), subsample=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (5, 5), padding='same', activation=LeakyReLU(0.1), subsample=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1, activation='simoid'))
        return model

    discrimnator_model=discrimnator_model()

    #generator 와 discriminator 를 Adam optimizer를 사용하여 컴파일한다.
    generator_model.compile(loss='binary_crosstropy',optimizer=Adam())
    discrimnator_model.compile(loss='binary_crossentropy',optimizer=Adam())

    generator_input = Input(shape=(100,))
    generator_output = generator_model(generator_input)
    discrimnator_model.trainable = False
    discriminator_output = discrimnator_model(generator_output)
    adversarial_model = Model(input=generator_input,output=discriminator_output)
    adversarial_model.summary()
    adversarial_model.compile(loss='binary_crossentropy',optimizer=Adam())

    #train 모듈 생성
    def train(epochs):
        batch_size = 128
        batch = 400
        for i in range(epochs):
            for j in tqdm(range(batch)):
                noise_1 = np.random.rand(batch_size,100)
                gen_images = generator_model.predict(noise_1,batch_size=batch_size)
                image_batch = X_train[np.random.randint(0,X_train.shape[0],size=batch_size)]
                disc_inp = np.concatenate([gen_images,image_batch])
                disc_Y = [0]*batch_size + [1]*batch_size
                discrimnator_model.trainable = True
                discrimnator_model.train_on_batch(disc_inp,disc_Y)

                noise_2 = np.random.randint(batch_size,100)
                discrimnator_model.trainable = False
                y_adv = [1]*batch_size
                adversarial_model.train_on_batch(noise_2,y_adv)
                train(80)

import matplotlib as plt

    def plot_output(text):
        try_input = np.random.rand(50,100)
        predictions = generator_model.predict(try_input)
        plt.figure(figsize=(20,20))

        for i in range(predictions.shape[0]):
            plt.subplot(10,10,i+1)
            plt.imshow(predictions[i,:,:,0],cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(text)
            plot_output(80)
                        





