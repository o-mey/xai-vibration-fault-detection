import os

import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler

np.random.seed(42)


def data_sine_time():
    def read_files(path, label):
        data = loadmat(path)
        data_x = data['data_ts']
        data_t = data['data_ts_t'].T
        data_rpm = data['data_ts_rpm']

        splits = 1000
        splitlen = len(data_x.T) // splits * splits
        split_x = np.array_split(data_x[0, :splitlen], splits)

        split_rpm = np.array_split(data_rpm[0, :splitlen], splits)
        rpm = np.array(split_rpm)

        X = np.expand_dims(np.array(split_x), -1)
        y = np.zeros((splits, 1)) + label
        return X, y, data_t[:len(X[0])], rpm

    X_cut, y_cut, t, rpm1 = read_files('./data_sin/sin_cutoff.mat', 1)
    X_uncut, y_uncut, _, rpm2 = read_files('./data_sin/sin.mat', 0)

    X = np.vstack([X_cut, X_uncut])
    y = np.vstack([y_cut, y_uncut])
    y_trafo = np.zeros((y.shape[0], 2))
    y_trafo[y[:, 0] == 0, 0] = 1
    y_trafo[y[:, 0] == 1, 1] = 1
    rpm = np.vstack([rpm1, rpm2])

    indices = [i for i in range(len(y))]
    indices_train = indices.copy()
    indices_test = indices[::5]
    del indices_train[::5]

    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train = y_trafo[indices_train]
    y_test = y_trafo[indices_test]
    rpm_train = rpm[indices_train]
    rpm_test = rpm[indices_test]

    return X_train, X_test, y_train, y_test, rpm_train, rpm_test, t


def data_sine_fft():
    def read_files(path, label):
        data = loadmat(path)
        data_x = data['data_fft_map'].T
        data_frequencies = data['data_fft_freq']
        data_rpm = data['rpmOut_fft']
        data_x = np.expand_dims(data_x, -1)

        X = data_x
        y = np.zeros((len(X), 1)) + label
        return X, y, data_frequencies, data_rpm

    X_cut, y_cut, f, rpm1 = read_files('./data_sin/sin_cutoff.mat', 1)
    X_uncut, y_uncut, _, rpm2 = read_files('./data_sin/sin.mat', 0)

    X = np.vstack([X_cut, X_uncut])
    y = np.vstack([y_cut, y_uncut])
    y_trafo = np.zeros((y.shape[0], 2))
    y_trafo[y[:, 0] == 0, 0] = 1
    y_trafo[y[:, 0] == 1, 1] = 1
    rpm = np.vstack([rpm1, rpm2])

    # path = './scalers/sinus_scaler_fft.pkl'
    # if os.path.isfile(path):
    #     scaler = joblib.load(path)
    # else:
    #     scaler = RobustScaler(quantile_range=(5, 95)).fit(X.reshape(-1, X.shape[1]))
    #     joblib.dump(scaler, path)
    # X = scaler.transform(X.reshape(-1, X.shape[1]))
    # X = np.expand_dims(X, -1)

    indices = [i for i in range(len(y))]
    indices_train = indices.copy()
    indices_test = indices[::5]
    del indices_train[::5]

    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train = y_trafo[indices_train]
    y_test = y_trafo[indices_test]
    rpm_train = rpm[indices_train]
    rpm_test = rpm[indices_test]
    return X_train, X_test, y_train, y_test, rpm_train, rpm_test, f


def data_sine_order():
    def read_files(path, label):
        data = loadmat(path)
        data_x = data['data_order_map'].T
        data_orders = data['data_order_orders']
        data_rpm = data['rpmOut_orders'].T
        data_x = np.expand_dims(data_x, -1)

        X = data_x
        y = np.zeros((len(X), 1)) + label
        return X, y, data_orders, data_rpm

    X_cut, y_cut, orders, rpm1 = read_files('./data_sin/sin_cutoff.mat', 1)
    X_uncut, y_uncut, _, rpm2 = read_files('./data_sin/sin.mat', 0)

    X = np.vstack([X_cut, X_uncut])
    y = np.vstack([y_cut, y_uncut])
    y_trafo = np.zeros((y.shape[0], 2))
    y_trafo[y[:, 0] == 0, 0] = 1
    y_trafo[y[:, 0] == 1, 1] = 1
    rpm = np.vstack([rpm1.T, rpm2.T])

    #path = './scalers/sinus_scaler_order.pkl'
    #if os.path.isfile(path):
    #    scaler = joblib.load(path)
    #else:
    #    scaler = RobustScaler(quantile_range=(5, 95)).fit(X.reshape(-1, X.shape[1]))
    #    joblib.dump(scaler, path)
    #X = scaler.transform(X.reshape(-1, X.shape[1]))
    #X = np.expand_dims(X, -1)

    indices = [i for i in range(len(y))]
    indices_train = indices.copy()
    indices_test = indices[::5]
    del indices_train[::5]

    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train = y_trafo[indices_train]
    y_test = y_trafo[indices_test]
    rpm_train = rpm[indices_train]
    rpm_test = rpm[indices_test]

    return X_train, X_test, y_train, y_test, rpm_train,rpm_test, orders


def data_fhg_time():
    def read_files(path, label):
        if label > 0:
            label = 1
        data = loadmat(path)
        data_x = data['data_ts']
        data_rpm = data['data_ts_rpm']

        split_len = 4096
        len_after_split = len(data_x) // split_len * split_len

        split_rpm = np.split(data_rpm[:len_after_split, 0], len_after_split / split_len)
        rpm = np.array(split_rpm)

        split_x = np.split(data_x[:len_after_split, 0], len_after_split / split_len)
        X = np.expand_dims(np.array(split_x), -1)
        y = np.zeros(shape=(len(X), 1)) + label
        return X, y, rpm

    X_D, X_E, y_D, y_E, rpm_D, rpm_E = [], [], [], [], [], []

    for i in range(5):
        X, y, rpm = read_files('./data_unwucht/' + str(i) + 'D.csv.mat', i)
        X_D.append(X)
        y_D.append(y)
        rpm_D.append(rpm)
        X, y, rpm = read_files('./data_unwucht/' + str(i) + 'E.csv.mat', i)
        X_E.append(X)
        y_E.append(y)
        rpm_E.append(rpm)

    X_train = np.vstack(X_D)
    X_test = np.vstack(X_E)
    y_train = np.vstack(y_D)
    y_test = np.vstack(y_E)
    rpm_train = np.vstack(rpm_D)
    rpm_test = np.vstack(rpm_E)

    y_trafo_train = np.zeros((y_train.shape[0], 2))
    y_trafo_train[y_train[:, 0] == 0, 0] = 1
    y_trafo_train[y_train[:, 0] == 1, 1] = 1

    y_trafo_test = np.zeros((y_test.shape[0], 2))
    y_trafo_test[y_test[:, 0] == 0, 0] = 1
    y_trafo_test[y_test[:, 0] == 1, 1] = 1
    return X_train, y_trafo_train, X_test, y_trafo_test, rpm_train, rpm_test


def data_fhg_fft():
    def read_files(path, label):
        if label > 0:
            label = 1
        data = loadmat(path)
        data_x = data['data_fft_map'].T
        data_frequencies = data['data_fft_freq']
        data_rpm = data['data_fft_rpm']
        data_x = np.expand_dims(data_x, -1)

        X = data_x
        y = np.zeros((len(X), 1)) + label
        return X, y, data_frequencies, data_rpm

    X_D, X_E, y_D, y_E, rpm_D, rpm_E = [], [], [], [], [], []
    for i in range(5):
        X, y, f, rpm = read_files('./data_unwucht/' + str(i) + 'D.csv.mat', i)
        X_D.append(X)
        y_D.append(y)
        rpm_D.append(rpm)
        X, y, _, rpm = read_files('./data_unwucht/' + str(i) + 'E.csv.mat', i)
        X_E.append(X)
        y_E.append(y)
        rpm_E.append(rpm)

    X_train = np.vstack(X_D)
    X_test = np.vstack(X_E)
    y_train = np.vstack(y_D)
    y_test = np.vstack(y_E)
    rpm_train = np.vstack(rpm_D)
    rpm_test = np.vstack(rpm_E)

    y_trafo_train = np.zeros((y_train.shape[0], 2))
    y_trafo_train[y_train[:, 0] == 0, 0] = 1
    y_trafo_train[y_train[:, 0] == 1, 1] = 1

    y_trafo_test = np.zeros((y_test.shape[0], 2))
    y_trafo_test[y_test[:, 0] == 0, 0] = 1
    y_trafo_test[y_test[:, 0] == 1, 1] = 1

    path = './scalers/unwucht_scaler_fft.pkl'
    if os.path.isfile(path):
        scaler = joblib.load(path)
    else:
        scaler = RobustScaler(quantile_range=(5, 95)).fit(X_train.reshape(-1, X_train.shape[1]))
        joblib.dump(scaler, path)
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[1]))
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1]))
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    return X_train, y_trafo_train, X_test, y_trafo_test, rpm_train, rpm_test, f


def data_fhg_order():
    def read_files(path, label):
        if label > 0:
            label = 1
        data = loadmat(path)
        data_x = data['data_order_map'].T
        data_orders = data['data_order_orders']
        data_rpm = data['data_order_rpm']
        data_x = np.expand_dims(data_x, -1)

        X = data_x
        y = np.zeros((len(X), 1)) + label
        return X, y, data_orders, data_rpm

    X_D, X_E, y_D, y_E, rpm_D, rpm_E = [], [], [], [], [], []
    for i in range(5):
        X, y, orders, rpm = read_files('./data_unwucht/' + str(i) + 'D.csv.mat', i)
        X_D.append(X)
        y_D.append(y)
        rpm_D.append(rpm)
        X, y, _, rpm = read_files('./data_unwucht/' + str(i) + 'E.csv.mat', i)
        X_E.append(X)
        y_E.append(y)
        rpm_E.append(rpm)

    orders_min_D = np.min([x.shape[1] for x in X_D])
    orders_min_E = np.min([x.shape[1] for x in X_E])
    orders_min = np.min([orders_min_D, orders_min_E])
    X_D = [x[:, :orders_min] for x in X_D]
    X_E = [x[:, :orders_min] for x in X_E]

    X_train = np.vstack(X_D)
    X_test = np.vstack(X_E)
    y_train = np.vstack(y_D)
    y_test = np.vstack(y_E)
    rpm_train = np.vstack(rpm_D)
    rpm_test = np.vstack(rpm_E)

    y_trafo_train = np.zeros((y_train.shape[0], 2))
    y_trafo_train[y_train[:, 0] == 0, 0] = 1
    y_trafo_train[y_train[:, 0] == 1, 1] = 1

    y_trafo_test = np.zeros((y_test.shape[0], 2))
    y_trafo_test[y_test[:, 0] == 0, 0] = 1
    y_trafo_test[y_test[:, 0] == 1, 1] = 1

    path = './scalers/unwucht_scaler_order.pkl'
    if os.path.isfile(path):
        scaler = joblib.load(path)
    else:
        scaler = RobustScaler(quantile_range=(5, 95)).fit(X_train.reshape(-1, X_train.shape[1]))
        joblib.dump(scaler, path)
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[1]))
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1]))
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return X_train, y_trafo_train, X_test, y_trafo_test, rpm_train, rpm_test, orders[:orders_min]


def array_for_pcolor(array):
    try:
        array = array.values
    except AttributeError:
        pass
    finally:
        array = np.array(array)
        if len(array) > 1:
            array_for_map = array.copy()
            if np.issubdtype(array.dtype, np.datetime64):
                d_array = (np.diff(array) / np.timedelta64(1, 's')).astype('timedelta64[s]')
            else:
                d_array = np.diff(array)
                array_for_map = np.float64(array_for_map)

            array_for_map[1:] = array_for_map[1:] - d_array / 2
            array_for_map[0] = array_for_map[0] - d_array[0] / 2
            array_for_map = np.append(array_for_map,
                                      array_for_map[len(array_for_map) - 1] + d_array[
                                          len(array_for_map) - 2])
        else:
            array_for_map = np.array([0.5 * array[0], 1.5 * array[0]]) \
                if array[0] != 0 else [-0.5, 0.5]
        return array_for_map


if __name__ == '__main__':
    def run_sine():
        X_train_time, X_test_time, y_train_time, y_test_time, rpm_time_train, rpm_time_test, time = data_sine_time()
        X_train_fft, X_test_fft, y_train_fft, y_test_fft, rpm_train_fft, rpm_test_fft, frequencies = data_sine_fft()
        X_train_orders, X_test_orders, y_train_orders, y_test_orders, rpm_train_orders, rpm_test_orders, orders = data_sine_order()

        print(X_train_time.shape, y_train_time.shape, rpm_time_train.shape, time.shape)
        print(X_train_fft.shape, y_train_fft.shape, rpm_train_fft.shape, frequencies.shape)
        print(X_train_orders.shape, y_train_orders.shape, rpm_train_orders.shape, orders.shape)


    def run_fhg():
        X_train, y_train, X_test, y_test, rpm_train, rpm_test = data_fhg_time()
        X_train_fft, y_train_fft, X_test_fft, y_test_fft, rpm_train_fft, rpm_test_fft, frequencies = data_fhg_fft()
        X_train_orders, y_train_orders, X_test_orders, y_test_orders, rpm_train_orders, rpm_test_orders, orders = data_fhg_order()

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print(X_train_fft.shape, y_train_fft.shape, X_test_fft.shape, y_test_fft.shape, frequencies.shape)
        print(X_train_orders.shape, y_train_orders.shape, X_test_orders.shape, y_test_orders.shape, orders.shape)


    run_sine()
    run_fhg()
