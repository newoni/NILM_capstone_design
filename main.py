import numpy as np
import pickle
import time

from lib.GAN import GAN
from lib.preprocessing import MinMax_scaler
from lib.preprocessing import MaxAbs_scaler
from lib.LSTM import LSTM
from lib.auto_encoder import AE
from lib.simple_classifier import Classification
import keras.models


if __name__ =="__main__":
    start = time.time()
    '''
    Data Load and Preprocessing
    data[0] = orginal data
    data[1] = LSTM prediction label
    data[2] = GAN. But 0 now
    data[3] = AE label. same with origin data
    data[4] = Classification lable 
    '''

    with open("data\\training_data_list_gap_0607.pickle",'rb') as fr:
        diffed_data = pickle.load(fr)

    with open("data\\training_data_list_no_gap_0607.pickle",'rb') as fr:
        origin_data = pickle.load(fr)

    sum_data = []
    origin_tmp = origin_data[0][:,0].reshape(-1,1)
    diffed_tmp = diffed_data[0][:,0].reshape(-1,1)
    buff_arr =np.concatenate([origin_tmp,diffed_tmp],axis=1)
    sum_data.append(buff_arr)

    for i in range(3):
        sum_data.append(0)
    sum_data.append(diffed_data[4])
    data = sum_data

    ####s <20.06.07> 데이터 분포를 더 넓게 하기 위해서 MaxAbs Scailer 사용
    # scaler = MinMax_scaler()
    scaler = MaxAbs_scaler()
    # scaler4_prediction = MinMax_scaler()
    scaler4_prediction = MaxAbs_scaler()
    ####e

    scaled_data_0 = scaler.fit_transform(data[0])
    # scaled_data_3 = scaler.sc.transform(data[3]) # AE model, same as data[0] (origin data)
    # scaled_data_1 = scaler4_prediction.fit_transform(data[1]) # data[2], data[4] don't need to preprocess cause it's value

    scaled_l = []
    scaled_l.append(scaled_data_0)
    scaled_l.append(0)
    scaled_l.append(0)
    scaled_l.append(0)
    scaled_l.append(data[4])
    
    # reshaped_data = scaled_data.reshape(scaled_data.shape[0],data.shape[1],1) #차원 맞춰주기 일단 대기 LSTM용이었음

    '''
    GAN
    '''

    # with open("data\\GAN_0604_ver2.pickle", 'rb') as fr:
    #     gan_data = pickle.load(fr)
    #
    # gan4data0_x = gan_data[0][0]
    # scaled_gan4data0_x = scaler.fit_transform(gan4data0_x)
    # gan4data0_y = gan_data[0][1]
    #
    # gan4data3_x = gan_data[3][0]
    # scaled_gan4data3_x = scaler.fit_transform(gan4data3_x)
    # gan4data3_y = gan_data[3][1]
    #
    # gan4data7_x = gan_data[7][0]
    # scaled_gan4data7_x = scaler.fit_transform(gan4data7_x)
    # gan4data7_y = gan_data[7][1]
    #
    # # gan4data0_x = scaled_gan4data0_x.reshape(np.shape(gan4data0_x)[0], np.shape(gan4data0_x)[1], 1)
    #
    # gan_model = GAN(scaled_gan4data0_x, epochs=50, batch_size=300, batch=5)
    # gan_model.iter_oper()
    #
    # gan_model3 = GAN(scaled_gan4data3_x, epochs=50, batch_size=300, batch=5)
    # gan_model3.iter_oper()
    #
    # gan_model7 = GAN(scaled_gan4data7_x, epochs=50, batch_size=300, batch=5)
    # gan_model7.iter_oper()
    #
    # cp_data = gan_model.cp_data().reshape(-1,10)
    # origin_data = scaled_gan4data0_x.reshape(-1, 10)
    #
    # cp_data3 = gan_model3.cp_data().reshape(-1, 10)
    # origin_data3 = scaled_gan4data3_x.reshape(-1, 10)
    #
    # cp_data7 = gan_model7.cp_data().reshape(-1, 10)
    # origin_data7 = scaled_gan4data7_x.reshape(-1, 10)

    '''
    LSTM data example
    '''
    train_data_X = np.arange(50*24)
    train_data_X = train_data_X.reshape(50,-1)
    # train_data_X = train_data_X.reshape(train_data_X.shape[0],24,1)
    train_data_Y = np.arange(50*1)
    train_data_Y = train_data_Y.reshape(50,-1)

    test_data_X = np.arange(24)
    test_data_X = test_data_X.reshape(1,24)
    # test_data_X = test_data_X.reshape(test_data_X.shape[0],24,1)

    # LSTM_model = LSTM()
    # LSTM_model.get_train_data(train_data_X, train_data_Y)
    # LSTM_model.get_test_data(train_data_X, train_data_Y)
    # LSTM_model.set_model()
    # LSTM_model.compile_fit_model()
    #
    # print(LSTM_model.model.predict(train_data_X))

    '''
    Auto encoder example
    '''
    x_nodes = 2
    z_dim = 1

    auto_encoder = AE(x_nodes, z_dim)
    history = auto_encoder.fit(scaled_l[0], scaled_l[0], epochs=300, batch_size=300, shuffle=True, validation_data=(scaled_l[0], scaled_l[0])) # Check shuffle 확인하기.

    encoder = auto_encoder.Encoder()
    decoder = auto_encoder.Decoder()

    encoded_imgs = encoder.predict(scaled_l[0])
    decoded_imgs = decoder.predict(encoded_imgs)        # 20.06.01 check. Min max scailing 필요.

    encoder.save('encoder.h5')
    decoder.save('decoder.h5')
    auto_encoder.save('auto_encoder.h5')

    '''
    Simple Classifier
    '''

    simple_classifier = Classification()
    simple_classifier.epoch = 10000
    simple_classifier.batch_size = 200

    simple_classifier.definition_model()
    simple_classifier.compile_model()
    simple_classifier.fit_model(decoded_imgs,scaled_l[4])
    result = simple_classifier.predict(scaled_data_0)   # 출력 단 수정 필요 One-hot encoding 필요
    cf = scaled_l[4]
    # simple_classifier.accuracy_model(test_x, test_y) # 현재 test 데이터가 없으므로 일단 주석 처리.

    # simple_classifier.predict(test_data_X)
    ###s gan 결과 비교
    # rere = simple_classifier.predict(cp_data)
    # rere3 = simple_classifier.predict(cp_data3)
    # rere7 = simple_classifier.predict(cp_data7)
    ####e

    simple_classifier.model.save('origin-diff_model.h5')

    # models.load_model('simple_classifier.h5')

    alpha = simple_classifier.predict(scaled_data_0)
    beta = data[4]

    print("소요시간: ", time.time() - start)
