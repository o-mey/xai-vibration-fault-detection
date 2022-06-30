clear all

cutoff = false;
start_time_second = 0;
end_time_second = 10*30;

fs_Hz = 500;
rot_frequency_min_rpm = 5*60;
rot_frequency_max_rpm = 10*60;

data_ts_t = start_time_second:1/fs_Hz:end_time_second;

rot_range_rpm = [rot_frequency_min_rpm, rot_frequency_max_rpm];
rot_samples_rpm = [start_time_second, end_time_second];
rot_freq_rpm = interp1(rot_samples_rpm, rot_range_rpm, data_ts_t);

data_ts_fs = fs_Hz;

data_ts = chirp(data_ts_t, rot_frequency_min_rpm/60.0, end_time_second, rot_frequency_max_rpm/60.0);
data_ts_const_1 = sin(data_ts_t*400);
data_ts_const_2 = sin(data_ts_t*500);

data_ts_rpm = rot_freq_rpm;

if cutoff == true
    data_ts(data_ts < -0.7) = -0.7;
end

figure(1)

data_ts = data_ts + data_ts_const_1 + data_ts_const_2;

plot(data_ts)
hold on;
plot(data_ts_rpm)
%plot(result_rpm_flatten)

figure(2)
[data_fft_map,data_fft_freq,rpmOut_fft,data_fft_time] = rpmfreqmap(data_ts, data_ts_fs, data_ts_rpm, 1);

imagesc(data_fft_time,data_fft_freq,data_fft_map)
ax = gca;
ax.YDir = 'normal';
xlabel('Time (s)')
ylabel('Frequency (Hz)')

figure(3)
[data_order_map,data_order_orders,rpmOut_orders,data_order_time] = rpmordermap(data_ts, data_ts_fs, data_ts_rpm, 0.25);
imagesc(data_order_time,data_order_orders,data_order_map)
ax = gca;
ax.YDir = 'normal';
xlabel('Time (s)')
ylabel('Order')

if cutoff == true
    save('../data_sin/sin_cutoff.mat', 'data_ts', 'data_ts_t','data_ts_fs', 'data_ts_rpm', 'data_fft_map', 'rpmOut_fft', 'data_fft_freq', 'data_fft_time', 'data_order_map', 'data_order_orders', 'rpmOut_orders', 'data_order_time');
else
    save('../data_sin/sin.mat',        'data_ts', 'data_ts_t','data_ts_fs', 'data_ts_rpm', 'data_fft_map', 'rpmOut_fft', 'data_fft_freq', 'data_fft_time', 'data_order_map', 'data_order_orders', 'rpmOut_orders', 'data_order_time');
end



