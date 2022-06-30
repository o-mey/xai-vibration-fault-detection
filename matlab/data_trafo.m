clear all
path = 'C:/Users/NED2ABT/Downloads/fraunhofer_eas_dataset_for_unbalance_detection_v1/';

dirs = ["0D.csv", "1D.csv", "2D.csv",  "4D.csv","0E.csv",...
        "2E.csv", "3E.csv", "4E.csv", "3D.csv", "1E.csv"];
for d = 1:size(dirs,2)
    cur_path = strcat('C:/Users/NED2ABT/Downloads/fraunhofer_eas_dataset_for_unbalance_detection_v1/', dirs(d))
    fprintf('fft')
    data0D = readtable(cur_path);
    skip = 50000;
    samples_per_second = 4096;
    data_ts_fs = 1.0/samples_per_second;
    seconds_per_analysis = 1.0;

    data_ts = data0D(skip:end,:).Vibration_1;
    data_ts_rpm = data0D(skip:end,:).Measured_RPM;
    data_ts_rpm(end) = data_ts_rpm(end-1); %fix nan values at end of rpm array
    [data_fft_map,data_fft_freq,data_fft_rpm,data_fft_time] = rpmfreqmap(data_ts, samples_per_second, data_ts_rpm, 1);

    figure(1)
    imagesc(data_fft_time,data_fft_freq,data_fft_map)
    ax = gca;
    ax.YDir = 'normal';
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    figure(2)
    fprintf('order')
    
    [data_order_map,data_order_orders,data_order_rpm,data_order_time] = rpmordermap(data_ts, samples_per_second, data_ts_rpm, 0.05);
    imagesc(data_order_time, data_order_orders, data_order_map)
    ax = gca;
    ax.YDir = 'normal';
    xlabel('Time (s)')
    ylabel('Order')

    save(strcat('../data_unwucht/', dirs(d), '.mat'), 'data_ts','data_ts_fs', 'data_ts_rpm', 'data_fft_map', 'data_fft_rpm', 'data_fft_freq', 'data_fft_time', 'data_order_map', 'data_order_rpm', 'data_order_orders', 'data_order_time');
end