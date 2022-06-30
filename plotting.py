import copy
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
from scipy.stats import median_abs_deviation
import matplotlib
import matplotlib as mpl
import tensorflow as tf
from tensorflow import divide

from datasets import data_sine_time, data_sine_fft, data_sine_order, data_fhg_time, data_fhg_fft, data_fhg_order

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Times New Roman'


def scale_gradcam(cam):
    extend = [np.min(cam), np.max(cam)]
    if np.sign(extend[0] * extend[1]) == -1:
        plusminus_ratio = 1 + np.min(np.abs(extend)) / np.max(np.abs(extend))
        cam_scaled = (cam) / (extend[1] - extend[0]) \
                     * plusminus_ratio
    else:
        cam_scaled = cam / np.max(np.abs(extend))
    return cam_scaled


def normalize2(x):
    """Utility function to normalize a tensor 99th quantile"""
    return x / (np.quantile(x, q=0.99) + (1e-10))


def plot_page(path, first_id0, first_id1, second_id0, second_id1, goal_path=None, x_values=None, run_name=None,
              y_values=None, x_axis_name="", class_names=None, data_paths=None, plot_names=None, figsize=None):
    fig, axes = plt.subplots(nrows=len(plot_names), ncols=2, figsize=figsize, dpi=300)
    axes_fl = list(axes.flat)
    for i in np.arange(len(plot_names)):
        print(plot_names[i])

        if plot_names[i] == 'Input':
            if 'sine' in path:
                quant_max = 1.0
            else:  # if run_name == 'fhg':
                quant_max = 0.95

        elif 'LIME' in plot_names[i]:
            quant_max = 0.95

        elif 'CAM' in plot_names[i]:
            quant_max = 0.95
        else:
            quant_max = 0.99
        quant_min = 0.0

        id = i * 2 + 0
        ax = axes_fl[id]
        if i == 0:
            ax.set_title(class_names[0])

        data = np.load(path + '/' + data_paths[i])
        data = np.squeeze(data)

        if 'CAM' in plot_names[i] or 'LRP' in plot_names[i] or 'Input' in plot_names[i]:
            results_line_normalized = np.zeros_like(data)
            for d in range(data.shape[0]):
                results_line_normalized[d, :] = normalize2(data[d, :].astype(np.float32))
            data = results_line_normalized

        if 'LIME' in plot_names[i]:
            plot_data = np.abs(data)[:data.shape[0] // 2, :]  # abs
        else:
            plot_data = np.maximum(data[first_id0:first_id1 - 1, :], 0)  # relu

        if 'LRP' in plot_names[i]:
            if 'sine' in path:
                plot_data[plot_data < 0.00001] = 0.00001
                norm = LogNorm(vmin=0.001, vmax=0.005)
            else:
                norm = LogNorm(vmin=0.01, vmax=0.8)
        else:
            min_quantile = np.quantile(plot_data, quant_min)
            max_quantile = np.quantile(plot_data, quant_max)
            norm = Normalize(vmin=min_quantile, vmax=max_quantile)

        ax.set_ylabel("Rot. Speed [rpm]")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:8.2f}"))
        ax.text(-0.27, 0.5, plot_names[i], rotation=90, transform=ax.transAxes, va='center', weight="bold")
        ax.pcolor(x_values[:, 0],
                      np.linspace(y_values[first_id0, 0], y_values[first_id1 - 1, 0],
                                  plot_data.shape[0]),
                      plot_data,
                      norm=norm,
                      cmap='viridis')

        if i == len(plot_names) - 1:
            ax.set_xlabel(x_axis_name)
            ax.get_xaxis().set_visible(True)
        else:
            ax.get_xaxis().set_ticklabels([])

        id = i * 2 + 1
        ax = axes_fl[id]

        if i == 0:
            ax.set_title(class_names[1])

        if 'LIME' in plot_names[i]:
            plot_data = np.abs(data)[data.shape[0] // 2:, :]
        else:
            plot_data = np.maximum(data[second_id0:second_id1 - 1, :], 0)

        if 'LRP' in plot_names[i]:
            if 'sine' in path:
                plot_data[plot_data < 0.000001] = 0.000001
                norm = LogNorm(vmin=0.001, vmax=0.005)
            else:
                print('lognorm')
                norm = LogNorm(vmin=0.01, vmax=0.8)

        if 'LRP' not in plot_names[i]:
            min_quantile = np.quantile(plot_data, quant_min)
            max_quantile = np.quantile(plot_data, quant_max)
            norm = Normalize(vmin=min_quantile, vmax=max_quantile)

        print(norm)
        im = ax.pcolor(x_values[:, 0],
                           np.linspace(y_values[second_id0, 0], y_values[second_id1 - 1, 0],
                                       plot_data.shape[0]),
                           plot_data,
                           norm=norm,
                           cmap='viridis')
        ax.get_yaxis().set_ticklabels([])
        if i == len(plot_names) - 1:
            ax.set_xlabel(x_axis_name)
            ax.get_xaxis().set_visible(True)
        else:
            ax.get_xaxis().set_ticklabels([])

    plt.tight_layout(h_pad=-2.5, w_pad=-1.5)
    plt.subplots_adjust(left=0.11, right=0.92, wspace=0.04, hspace=0.08)

    if len(plot_names) == 5:
        cax = fig.add_axes([0.93, 0.075, 0.02, 0.88])  # [left, bottom, width, height]
        cbar = plt.colorbar(im, cax=cax, ticks=[min_quantile, max_quantile])
        cbar.ax.set_yticklabels(['Low', 'High'])
        '''
        sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm)
        '''

    if len(plot_names) == 8:
        cax = fig.add_axes([0.93, 0.05, 0.02, 0.92])  # [left, bottom, width, height]
        cbar = plt.colorbar(im, cax=cax, ticks=[min_quantile, max_quantile])
        cbar.ax.set_yticklabels(['Low', 'High'])
        '''
        sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm)
        '''


    elif len(plot_names) == 4:
        cax = fig.add_axes([0.93, 0.09, 0.02, 0.855])
        cbar = plt.colorbar(im, cax=cax, ticks=[min_quantile, max_quantile])
        cbar.ax.set_yticklabels(['Low', 'High'])

    #plt.show()
    plt.savefig(goal_path + run_name + 'page.png')
    plt.close()


def plot_dataset_sine(data_paths, plot_names, figsize, goal_path, run_name):
    # X_train_time, X_test_time, y_train_time, y_test_time, rpm_time_train, rpm_time_test, time = data_sine_time()
    X_train_fft, X_test_fft, y_train_fft, y_test_fft, rpm_train_fft, rpm_test_fft, frequencies = data_sine_fft()
    X_train_orders, X_test_orders, y_train_orders, y_test_orders, rpm_train_orders, rpm_test_orders, orders = data_sine_order()

    plot_page('plotting/data/sine/fft_cnn', goal_path=goal_path + 'fft_', run_name=run_name,
              x_values=frequencies, y_values=rpm_test_fft,
              first_id0=0, first_id1=80, second_id0=80, second_id1=159,
              x_axis_name='Frequency [Hz]',
              class_names=['Cut-Off', 'Normal'],
              data_paths=data_paths, plot_names=plot_names, figsize=figsize)

    plot_page('plotting/data/sine/order_cnn', goal_path=goal_path + 'order_', run_name=run_name,
              x_values=orders, y_values=rpm_test_orders,
              first_id0=0, first_id1=60, second_id0=60, second_id1=119,
              x_axis_name='Orders',
              class_names=['Cut-Off', 'Normal'],
              data_paths=data_paths, plot_names=plot_names, figsize=figsize)


def plot_dataset_fhg(data_paths, plot_names, figsize, goal_path, run_name):
    # X_train_time, y_train, X_test, y_test, rpm_train, rpm_test = data_fhg_time()
    # time = np.arange(len(X_train_time[0])) * 0.001
    X_train_fft, y_train_fft, X_test_fft, y_test_fft, rpm_train_fft, rpm_test_fft, frequencies = data_fhg_fft()
    X_train_orders, y_train_orders, X_test_orders, y_test_orders, rpm_train_orders, rpm_test_orders, orders = data_fhg_order()

    plot_page('plotting/data/fhg/fft_cnn', goal_path=goal_path + 'fft_', run_name=run_name,
              x_values=frequencies, y_values=rpm_test_fft,
              first_id0=1165, first_id1=2150, second_id0=10017, second_id1=11134,
              x_axis_name='Frequency [Hz]',
              class_names=['Without Imbalance', 'With Imbalance'],
              data_paths=data_paths, plot_names=plot_names, figsize=figsize)

    plot_page('plotting/data/fhg/order_cnn', goal_path=goal_path + 'order_', run_name=run_name,
              x_values=orders, y_values=rpm_test_orders,
              first_id0=560, first_id1=1040, second_id0=4948, second_id1=5499,
              x_axis_name='Orders',
              class_names=['Without Imbalance', 'With Imbalance'],
              data_paths=data_paths, plot_names=plot_names, figsize=figsize)


if __name__ == '__main__':
    data_paths_all = ['Input.npy',

                      'lime_emplace.npy',
                      'lime_mean.npy',
                      'lime_total_mean.npy',
                      'lime_noise.npy',
                      'lime_total_noise.npy'
                      ]

    plot_names_all = ['Input',

                      'LIME (Global)',
                      'LIME (Mean)',
                      'LIME (Total Mean)',
                      'LIME (Noise)',
                      'LIME (Total Noise)'
                      ]

    figsize_all = (8.27, 11.69)

    plot_dataset_sine(data_paths_all, plot_names_all, figsize_all, goal_path='./plotting/plots/sine/',
                     run_name='lime_all')
    plot_dataset_fhg(data_paths_all, plot_names_all, figsize_all, goal_path='./plotting/plots/fhg/',
                    run_name='lime_all')

    data_paths_all = ['Input.npy',
                      'gradcam_pp.npy',
                      'scorecam.npy',
                      'LRPEpsilon.npy',
                      'Deconvnet.npy',
                      'BoundedDeepTaylor.npy',
                      'IntegratedGradients.npy',
                      'lime_total_noise.npy'
                      ]

    plot_names_all = ['Input',
                      'GradCAM++',
                      'ScoreCAM',
                      'LRP-Epsilon',
                      'Deconv-Net',
                      'Deep Taylor',
                      'Integrated Gradients',
                      'LIME (Total Noise)'
                      ]
    figsize_all = (8.27, 11.69)

    plot_dataset_sine(data_paths_all, plot_names_all, figsize_all, goal_path='./plotting/plots/sine/', run_name='all')
    plot_dataset_fhg(data_paths_all, plot_names_all, figsize_all, goal_path='./plotting/plots/fhg/', run_name='all')

    data_paths_all = ['Input.npy',
                      'gradcam_pp.npy',
                      'scorecam.npy',
                      'LRPEpsilon.npy',
                      'lime_total_noise.npy'
                      ]

    plot_names_all = ['Input',
                      'GradCAM++',
                      'ScoreCAM',
                      'LRP-Epsilon',
                      'LIME (Total Noise)'
                      ]
    figsize_all = (8.27, 7.65)

    plot_dataset_sine(data_paths_all, plot_names_all, figsize_all, goal_path='./plotting/plots/sine/', run_name='si')
    plot_dataset_fhg(data_paths_all, plot_names_all, figsize_all, goal_path='./plotting/plots/fhg/', run_name='si')

    data_paths_all = ['Input.npy',
                      'gradcam.npy',
                      'LRPZ.npy',
                      'lime_emplace.npy'
                      ]

    plot_names_all = ['Input',
                      'GradCAM',
                      'LRP-Z',
                      '(Global) LIME'
                      ]
    figsize_all = (8.27, 6.29)

    plot_dataset_sine(data_paths_all, plot_names_all, figsize_all, goal_path='./plotting/plots/sine/', run_name='paper')
    plot_dataset_fhg(data_paths_all, plot_names_all, figsize_all, goal_path='./plotting/plots/fhg/', run_name='paper')
