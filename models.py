import numpy as np
import tensorflow as tf
from keras.backend import floatx, categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, BatchNormalization, Dropout, Activation
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from tensorflow import convert_to_tensor
from tensorflow.python.framework.smart_cond import smart_cond
from tensorflow.python.ops import math_ops, array_ops

import datasets

# from dataset_sine_time import plot_time_series_grid

# code adapted from https://github.com/albermax/innvestigate/blob/feature/version2.0_rc0_branched/examples/notebooks/mnist_compare_methods.ipynb

split = 0.2
metric = 'val_loss'
n_epochs = 150


def categorical_crossentropy_new(from_logits=False, label_smoothing=0.0):
    def func(y_pred, y_true):
        nonlocal from_logits, label_smoothing

        def _smooth_labels():
            num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        y_pred = convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_smoothing = convert_to_tensor(label_smoothing, dtype=floatx())
        y_true = smart_cond(label_smoothing, _smooth_labels, lambda: y_true)
        return categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

    return func


def get_model_cnn(X_train, y_train, name, data_format, n_conv_layers, train=True, ):  # (None, len, 1)

    best_model_filepath = f"./models/" + name + "_get_model_" + data_format + "_cnn.h5"

    # n_conv_layers = 3  # [1,2,3,4]
    n_dense_units = 128
    dropout_rate = 0.0
    use_batch_normalization = True  # [True, False]
    filter_size = 9  # [5,7,9]
    learning_rate = 0.0001
    n_epochs = 150  # [50,100,200]

    weight_for_0 = np.sum(y_train[:, 1]) / np.sum(y_train)
    weight_for_1 = np.sum(y_train[:, 0]) / np.sum(y_train)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('class_weight:', class_weight)

    if train:
        X_in = Input(shape=(X_train.shape[1], 1), name="vibration_input")
        x = X_in
        for j in range(n_conv_layers):
            print(j)
            if j == (n_conv_layers - 1):
                x = Conv1D(filters=(j + 1) * 10,
                           kernel_size=filter_size,
                           strides=1,
                           padding="same",
                           activation='relu',
                           kernel_initializer='he_uniform', name="cam_layer")(x)
            else:
                x = Conv1D(filters=(j + 1) * 10,
                           kernel_size=filter_size,
                           strides=1,
                           padding="same",
                           activation='relu',
                           kernel_initializer='he_uniform')(x)
            if use_batch_normalization:
                x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
        x = Flatten()(x)
        x = Dense(units=n_dense_units, activation='relu')(x)
        # x = ReLU()(x)
        x = Dropout(rate=dropout_rate)(x)
        x = Dense(units=2, activation='linear', kernel_regularizer=l1_l2(l1=0.0, l2=5e3))(x)
        X_out = Activation(tf.nn.softmax)(x)

        classifier = Model(X_in, X_out)

        checkpoint = ModelCheckpoint(best_model_filepath, monitor=metric,
                                     verbose=1, save_best_only=True, mode='auto')

        classifier.compile(optimizer=Adam(lr=learning_rate),
                           loss=categorical_crossentropy_new(from_logits=False, label_smoothing=0.1),
                           metrics=['accuracy'])

        classifier.fit(X_train, y_train, epochs=n_epochs, batch_size=16, verbose=2, class_weight=class_weight,
                       validation_split=split, callbacks=[checkpoint])

    classifier = load_model(best_model_filepath, custom_objects={
        'func': categorical_crossentropy_new(from_logits=False, label_smoothing=0.1)})
    classifier.summary()
    return classifier


if __name__ == '__main__':
    def test_sine():
        name = 'sine'
        X_train_time, X_test_time, y_train_time, y_test_time, rpm_time_train, rpm_time_test, time = datasets.data_sine_time()
        X_train_fft, X_test_fft, y_train_fft, y_test_fft, rpm_train_fft, rpm_test_fft, frequencies = datasets.data_sine_fft()
        X_train_orders, X_test_orders, y_train_orders, y_test_orders, rpm_train_orders, rpm_test_orders, orders = datasets.data_sine_order()

        print(X_train_time.shape)
        print(y_train_time.shape)
        print(X_train_fft.shape)

        model = get_model_cnn(X_train_time, y_train_time, name=name, data_format='time', n_conv_layers=3, train=True)
        score = model.evaluate(X_test_time, y_test_time, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # fft cnn
        model = get_model_cnn(X_train_fft, y_train_fft, data_format='fft', name=name, n_conv_layers=2, train=True)
        score = model.evaluate(X_test_fft, y_test_fft, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # order cnn
        model = get_model_cnn(X_train_orders, y_train_orders, data_format='order', name=name, n_conv_layers=2,
                              train=True)

        score = model.evaluate(X_test_orders, y_test_orders, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


    def test_fhg():
        name = 'fhg'
        X_train, y_train, X_test, y_test, rpm_train, rpm_test = datasets.data_fhg_time()
        X_train_fft, y_train_fft, X_test_fft, y_test_fft, rpm_train, rpm_test, f = datasets.data_fhg_fft()
        X_train_orders, y_train_orders, X_test_orders, y_test_orders, rpm_train, rpm_test, orders = datasets.data_fhg_order()

        model = get_model_cnn(X_train, y_train, name=name, data_format='time', n_conv_layers=3, train=False)
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # fft fcn vs cnn
        model = get_model_cnn(X_train_fft, y_train_fft, data_format='fft', name=name, n_conv_layers=2, train=False)
        score = model.evaluate(X_test_fft, y_test_fft, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # order fcn vs cnn
        model = get_model_cnn(X_train_orders, y_train_orders, data_format='order', name=name, n_conv_layers=2,
                              train=False)
        score = model.evaluate(X_test_orders, y_test_orders, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


    test_sine()
    #test_fhg()
