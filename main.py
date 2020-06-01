import numpy as np

# from lib.GAN import GAN
from lib.LSTM import LSTM
from lib.auto_encoder import AE


if __name__ =="__main__":
    '''
    LSTM data example
    '''
    train_data_X = np.arange(50*24)
    train_data_X = train_data_X.reshape(50,-1)
    train_data_X = train_data_X.reshape(train_data_X.shape[0],24,1)
    train_data_Y = np.arange(50*1)
    train_data_Y = train_data_Y.reshape(50,-1)

    test_data_X = np.arange(24)
    test_data_X = test_data_X.reshape(1,24)
    test_data_X = test_data_X.reshape(test_data_X.shape[0],24,1)

    LSTM_model = LSTM()
    LSTM_model.get_train_data(train_data_X, train_data_Y)
    LSTM_model.get_test_data(train_data_X, train_data_Y)
    LSTM_model.set_model()
    LSTM_model.compile_fit_model()

    print(LSTM_model.model.predict(train_data_X))

    '''
    Auto encoder example
    '''
    train4AE = LSTM_model.model.predict(train_data_X)