from matplotlib.colors import Normalize
from scipy.io import loadmat

import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import resample

from datasets import data_sine_fft, data_sine_order

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['font.size'] = 6
import plotting_preamble as pp
import numpy as np

if __name__ == '__main__':
    data = loadmat('./data_sin/sin_plot.mat')
    data_x = data['data_ts'].T
    data_x_cut = data['data_ts_cut'].T
    data_t = data['data_ts_t'].T

    data_x_const_1 = data['data_ts_const_1'].T
    data_x_const_2 = data['data_ts_const_2'].T

    count = 200
    start = 50850
    end = 51000

    X_train_fft, X_test_fft, y_train_fft, y_test_fft, rpm_train_fft, rpm_test_fft, frequencies = data_sine_fft(
        balance=False)
    X_train_orders, X_test_orders, y_train_orders, y_test_orders, rpm_train_orders, rpm_test_orders, orders = data_sine_order()

    f = plt.figure(figsize=(7.27, 5.5), dpi=300)
    ax01 = f.add_subplot(423)
    ax01.set_ylabel('Amplitude')
    x1, t = resample(np.sin(data_t[start:end] * 400), len(data_t[start:end]) * 4, t=data_t[start:end], axis=0,
                     window=None, domain='time')
    ax01.plot(t, x1)
    # ax01.plot(data_t[start:end], data_x_const_1[start:end])
    ax01.title.set_text('Addition 1')
    ax01.grid(True)
    ax01.set_xlabel('Time [s]')

    ax02 = f.add_subplot(424)
    x2, t = resample(np.sin(data_t[start:end] * 500), len(data_t[start:end]) * 4, t=data_t[start:end], axis=0,
                     window=None, domain='time')
    ax02.plot(t, x2, label='Addition 2')

    ax02.title.set_text('Addition 2')
    ax02.grid(True)
    ax02.yaxis.set_ticklabels([])
    ax02.set_xlabel('Time [s]')

    ax03 = f.add_subplot(421)
    x, t = resample(data_x[start:end], len(data_t[start:end]) * 4, t=data_t[start:end], axis=0,
                    window=None, domain='time')
    x_cut, t = resample(data_x_cut[start:end], len(data_t[start:end]) * 4, t=data_t[start:end], axis=0,
                        window=None, domain='time')
    ax03.plot(t, x + x1 + x2,
              )
    ax03.plot(t, x_cut + x1 + x2, linestyle='dashed')

    ax03.title.set_text('Sine Dataset')
    ax03.grid(True)
    ax03.set_ylabel('Amplitude')
    ax03.set_xlabel('Time [s]')

    ax00 = f.add_subplot(422)  # , figsize=(3.5,2.5), dpi=300, constrained_layout=True)
    ax00.plot(data_t[start:end], data_x[start:end], label='Normal')
    ax00.plot(data_t[start:end], data_x_cut[start:end], label='Cut-off', linestyle='dashed')
    ax00.legend(loc='upper right')
    ax00.set_ylim()
    ax00.title.set_text('Base Signal')
    ax00.set_ylim([-2.7, 2.7])
    ax00.yaxis.set_ticklabels([])
    ax00.grid(True)
    ax00.set_xlabel('Time [s]')

    first_id0 = 0
    first_id1 = 80
    second_id0 = 80
    second_id1 = 159
    ax04 = f.add_subplot(425)
    quant_max = 0.98
    max_quantile = np.quantile(X_test_fft, quant_max)
    norm = Normalize(vmin=0, vmax=max_quantile)
    ax04.pcolor(frequencies[:, 0],
                np.linspace(rpm_test_fft[first_id0, 0], rpm_test_fft[first_id1 - 1, 0],
                            X_test_fft[first_id0:first_id1, :, 0].shape[0]),
                X_test_fft[first_id0:first_id1, :, 0],
                norm=norm,
                cmap='viridis')
    ax04.set_ylabel('RPM')
    ax04.set_xlabel('Frequency [Hz]')
    ax04.title.set_text('Fourier Transform (Cut-off)')

    ax05 = f.add_subplot(426)
    ax05.pcolor(frequencies[:, 0],
                np.linspace(rpm_test_fft[second_id0, 0], rpm_test_fft[second_id1 - 1, 0],
                            X_test_fft[second_id0:second_id1, :, 0].shape[0]),
                X_test_fft[second_id0:second_id1, :, 0],
                norm=norm,
                cmap='viridis')
    ax05.yaxis.set_visible(False)
    ax05.set_xlabel('Frequency [Hz]')
    ax05.title.set_text('Fourier Transform (Normal)')
    # order
    first_id0 = 0
    first_id1 = 59
    second_id0 = 60
    second_id1 = 119


    ax06 = f.add_subplot(427)
    max_quantile = np.quantile(X_test_orders, quant_max)
    norm = Normalize(vmin=0, vmax=max_quantile)
    ax06.pcolor(orders[:, 0],
                np.linspace(rpm_test_orders[first_id0, 0], rpm_test_orders[first_id1 - 1, 0],
                            X_test_orders[first_id0:first_id1, :, 0].shape[0]),
                X_test_orders[first_id0:first_id1, :, 0],
                norm=norm,
                cmap='viridis')
    ax06.set_ylabel('RPM')
    ax06.set_xlabel('Orders')
    ax06.title.set_text('Order Analysis (Cut-off)')

    ax07 = f.add_subplot(428)
    ax07.yaxis.set_visible(False)

    ax07.pcolor(orders[:, 0],
                np.linspace(rpm_test_orders[second_id0, 0], rpm_test_orders[second_id1 - 1, 0],
                            X_test_orders[second_id0:second_id1, :, 0].shape[0]),
                X_test_orders[second_id0:second_id1, :, 0],
                norm=norm,
                cmap='viridis')

    ax07.set_xlabel('Orders')
    ax07.title.set_text('Order Analysis (Normal)')
    plt.tight_layout()
    # plt.show()

    # plt.savefig('../images/xai/sine_for_svg.png')
    plt.savefig('../images/xai/sine.png')
