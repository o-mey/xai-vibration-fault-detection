import glob
import math
import os
import re

import numpy as np

import innvestigate_master.utils as iutils
import models
from datasets import data_fhg_time, data_fhg_fft, data_fhg_order, data_sine_time, data_sine_fft, data_sine_order
from innvestigate_master import create_analyzer
from lime_for_time.lime_timeseries_batch_cnn_emplace import LimeTimeSeriesExplainer


# from matplotlib.colors import SymLogNorm


def input_postprocessing(X):  # for the time series case, postprocessing not used
    return X


methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

    # Show input
    ("input", {}, input_postprocessing, "Input"),
    ("gradient", {}, input_postprocessing, "Gradient"),
    ("smoothgrad", {}, input_postprocessing, "SmoothGrad"),

    # Signal
    ("deconvnet", {}, input_postprocessing, "Deconvnet"),
    ("guided_backprop", {}, input_postprocessing, "Guided Backprop",),

    # Interaction

    ("deep_taylor.bounded", {"low": 0,
                             "high": 1}, input_postprocessing, "DeepTaylor"),
    ("input_t_gradient", {}, input_postprocessing, "Input * Gradient"),
    ("integrated_gradients", {}, input_postprocessing, "Integrated Gradients"),

    ("lrp.z", {}, input_postprocessing, "LRP-Z"),
    ("lrp.epsilon", {}, input_postprocessing, "LRP-Epsilon"),
]


def get_analyzers(X_test, model_wo_softmax, batch_size=256):
    analyzers = []
    for method in methods:
        # print(method[0])
        analyzer = create_analyzer(method[0],  # analysis method identifier
                                   model_wo_softmax,  # model_wo_softmax, model without softmax output
                                   **method[1], neuron_selection_mode='index')  # optional analysis parameters

        # Some analyzers require training.
        analyzer.fit(X_test[0], batch_size=batch_size, verbose=1)
        analyzers.append(analyzer)
    return analyzers


def prepare_grid(analysis, text):
    grid = [[analysis[k, i, j] for j in range(analysis.shape[2])]
            for i in range(analysis.shape[1]) for k in range(analysis.shape[0])]
    # Prepare the labels
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('label: {}'.format(label[i]), 'neuron: {}'.format(pred[i])) for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
    col_labels = [''.join(method[3]) for method in methods]
    return col_labels, grid, row_labels_left, row_labels_right


def analyze_line(data, model, mode, n=4, max_len=-1, path=''):
    print('PATH:', path)
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    X_test, y_test = data
    analyzers = get_analyzers(X_test, model_wo_softmax)

    # Predict final activations, probabilites, and label.
    presm = model_wo_softmax.predict(X_test, batch_size=12)
    prob = model.predict(X_test, batch_size=12)
    y_hat = prob.argmax(axis=1)
    result = np.zeros_like(X_test)

    for aidx, analyzer in enumerate(analyzers):
        # Analyze.
        a_name = re.split(r"[\W']+", str(analyzer.__class__))[-2]

        os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
        for i in range(len(X_test)):
            a = analyzer.analyze(X_test[i:i + 1], neuron_selection=y_hat[
                                                                   i:i + 1])  # analyzer.analyze(X_test, neuron_selection=y_hat) # neuron_index
            result[i] = a
        np.save(path + '/' + a_name, result)
    np.save(path + '/' + 'y_hat', y_hat)
    np.save(path + '/' + 'y_test', y_test)
    np.save(path + '/' + 'X_test', X_test)


def run_lime(X_orders, y, classifier, num_features, num_slices, path, method):
    pred = classifier.predict(X_orders, batch_size=16)
    series = X_orders[pred[:, 0] <= 0.5]
    emplace = X_orders[pred[:, 0] > 0.5]

    explainer = LimeTimeSeriesExplainer(class_names=['1', '2'])
    exp = explainer.explain_instance(series, other_instance=emplace, classifier_fn=classifier.predict,
                                     num_features=num_features, num_samples=2500,
                                     labels=(1,),
                                     num_slices=num_slices,
                                     replacement_method=method)

    values_per_slice = math.ceil(len(series[0]) / num_slices)

    feature_weights = exp.as_list()
    features = [x for x, y in feature_weights]
    weights = np.array([y for x, y in feature_weights])
    max_weight = np.max(np.abs(weights))
    weights = weights / max_weight

    vals = list(zip(features, weights))
    result = np.zeros_like(series)
    for i in vals:
        feature, weight = i
        start = feature * values_per_slice
        end = start + values_per_slice
        result[:, start:end, 0] += weight
    os.makedirs(os.path.dirname(path + '/' + 'lime/' + method + '/'), exist_ok=True)
    np.save(path + '/' + 'lime/' + method + '/feat' + str(num_features) + '_slices' + str(num_slices) + '_class0',
            result)

    # reverse
    exp = explainer.explain_instance(emplace, other_instance=series, classifier_fn=classifier.predict,
                                     num_features=num_features, num_samples=2500, labels=(1,),
                                     num_slices=num_slices,
                                     replacement_method=method)

    values_per_slice = math.ceil(len(series[0]) / num_slices)

    feature_weights = exp.as_list()
    features = [x for x, y in feature_weights]
    weights = np.array([y for x, y in feature_weights])
    max_weight = np.max(np.abs(weights))
    weights = weights / max_weight

    vals = list(zip(features, weights))
    result = np.zeros_like(series)
    for i in vals:
        feature, weight = i
        start = feature * values_per_slice
        end = start + values_per_slice
        result[:, start:end, 0] += weight
    os.makedirs(os.path.dirname(path + '/' + 'lime/' + method + '/'), exist_ok=True)
    np.save(path + '/' + 'lime/' + method + '/feat' + str(num_features) + '_slices' + str(num_slices) + '_class1',
            result)


features_all = [3, 7, 10, 12, 15]
slices_all = [15, 20, 30, 50]
lime_methods = ['emplace', 'mean', 'total_mean', 'noise', 'total_noise']

if __name__ == '__main__':
    def run_sine():
        name = 'sine'
        # X_train_time, X_test_time, y_train_time, y_test_time, rpm_time_train, rpm_time_test, time = data_sine_time()
        X_train_fft, X_test_fft, y_train_fft, y_test_fft, rpm_train_fft, rpm_test_fft, frequencies = data_sine_fft()
        X_train_orders, X_test_orders, y_train_orders, y_test_orders, rpm_train_orders, rpm_test_orders, orders = data_sine_order()

        # print(X_train_time.shape, y_train_time.shape, rpm_time_train.shape, time.shape)
        print(X_train_fft.shape, y_train_fft.shape, rpm_train_fft.shape, frequencies.shape)
        print(X_train_orders.shape, y_train_orders.shape, rpm_train_orders.shape, orders.shape)

        # fft
        model = models.get_model_cnn(X_test_fft, y_test_fft, name=name, data_format='fft', n_conv_layers=2, train=False)
        analyze_line((X_test_fft, y_test_fft), model, mode='cnn', path='./plotting/data/sine/fft_cnn')

        for m in lime_methods:
            for f in features_all:
                for slices in slices_all:
                    run_lime(X_test_fft[::2], y_test_fft[::2], model, num_features=f, num_slices=slices,
                             path='./plotting/data/sine/fft_cnn', method=m)
        # orders
        model = models.get_model_cnn(X_test_orders, y_test_orders, name='sine', data_format='order', n_conv_layers=2,
                                     train=False)
        analyze_line((X_test_orders, y_test_orders), model, mode='cnn', path='./plotting/data/sine/order_cnn')
        for m in lime_methods:
            for f in features_all:
                for slices in slices_all:
                    run_lime(X_test_orders[::2], y_test_orders[::2], model, num_features=f, num_slices=slices,
                             path='./plotting/data/sine/order_cnn', method=m)


    def run_fhg_new():
        name = 'fhg'
        # X_train, y_train, X_test, y_test, rpm_train, rpm_test = data_fhg_time()
        X_train_fft, y_train_fft, X_test_fft, y_test_fft, rpm_train_fft, rpm_test_fft, frequencies = data_fhg_fft()
        X_train_orders, y_train_orders, X_test_orders, y_test_orders, rpm_train_orders, rpm_test_orders, orders = data_fhg_order()


        # fft + cnn
        model = models.get_model_cnn(X_train_fft, y_train_fft, name=name, data_format='fft', n_conv_layers=3,
                                     train=False)
        analyze_line((X_test_fft, y_test_fft), model, mode='cnn', path='./plotting/data/fhg/fft_cnn')

        # fft + lime
        for m in lime_methods:
            for f in features_all:
                for slices in slices_all:
                    run_lime(X_test_fft[::20], y_test_fft[::20], model, num_features=f, num_slices=slices,
                             path='./plotting/data/fhg/fft_cnn/', method=m)

        # orders + cnn
        model = models.get_model_cnn(X_test_orders, y_test_orders, name=name, data_format='order', n_conv_layers=3,
                                     train=False)
        analyze_line((X_test_orders, y_test_orders), model, mode='cnn', path='./plotting/data/fhg/order_cnn')

        # orders + lime
        for m in lime_methods:
            for f in features_all:
                for slices in slices_all:
                    run_lime(X_test_orders[::20], y_test_orders[::20], model, num_features=f, num_slices=slices,
                             path='./plotting/data/fhg/order_cnn/', method=m)  # [::5]


    def merge_lime(path):
        for m in lime_methods:

            lime_files = glob.glob(path + '/' + 'lime/' + m + '/feat*_slices*_class0.npy')
            class0 = []
            class1 = []
            for i_f, file_name in enumerate(lime_files):
                # do lime
                data_lime = np.load(file_name)
                class0.append(data_lime.copy())

                class1_path = file_name.replace('class0', 'class1')
                data_lime = np.load(class1_path)
                class1.append(data_lime.copy())

            # mean lime data plot
            class0_mean = np.mean(np.array(class0), axis=0)[:, :, 0]
            class1_mean = np.mean(np.array(class1), axis=0)[:, :, 0]
            class_both = np.vstack([class0_mean, class1_mean])
            np.save(path + 'lime_' + m, class_both)

    run_sine()
    run_fhg_new()

    merge_lime('./plotting/data/fhg/order_cnn/')
    merge_lime('./plotting/data/fhg/fft_cnn/')

    merge_lime('./plotting/data/sine/fft_cnn/')
    merge_lime('./plotting/data/sine/order_cnn/')
