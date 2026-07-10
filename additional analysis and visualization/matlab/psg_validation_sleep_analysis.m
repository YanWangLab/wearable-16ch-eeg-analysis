function psg_validation_sleep_analysis()
clear; clc; close all force;

%% Run switches

RUN_PREPROCESS_WIRELESS_SLEEP_EEG = false;  % regenerates eeg_data_filtered.mat from raw_data.mat
RUN_OVERNIGHT_VT_STFT             = true;   % whole-night raw traces + panel-b-style PSG/wireless spectrogram
RUN_DOCTOR_CHECK_30S_EXPORT       = false;  % exports 30 s epoch images; very large output
RUN_REPRESENTATIVE_30S_WAVEFORM   = true;   % representative stacked waveform, seconds x-axis
RUN_WHOLE_NIGHT_STACKED_WAVEFORM  = false;  % already generated inside RUN_OVERNIGHT_VT_STFT
RUN_SLEEP_STAGE_AGREEMENT         = true;   % hypnogram/confusion-matrix source metrics


%% Paths

DATA_ROOT = fullfile('example_data', 'psg_validation');
OUTPUT_ROOT = fullfile('outputs', 'psg_validation');

RAW_DATA_FILE        = fullfile(DATA_ROOT, 'raw_data.mat');
WIRELESS_FILTER_FILE = fullfile(DATA_ROOT, 'eeg_data_filtered.mat');
HOSPITAL_EEG_FILE    = fullfile(DATA_ROOT, 'eeg_data_hospital.mat');
SLEEP_STAGE_FILE     = fullfile(DATA_ROOT, 'sleep_stage_labels.xlsx');

OUT_VT_DIR      = fullfile(OUTPUT_ROOT, 'time_domain');
OUT_STFT_DIR    = fullfile(OUTPUT_ROOT, 'spectrogram');
OUT_DOCTOR_DIR  = fullfile(OUTPUT_ROOT, 'doctor_check_epochs');
OUT_STAGE_DIR   = fullfile(OUTPUT_ROOT, 'sleep_stage_agreement');

ensure_dir(OUTPUT_ROOT);
ensure_dir(OUT_VT_DIR);
ensure_dir(OUT_STFT_DIR);
ensure_dir(OUT_DOCTOR_DIR);
ensure_dir(OUT_STAGE_DIR);


%% Run selected analysis parts

if RUN_PREPROCESS_WIRELESS_SLEEP_EEG
    preprocess_wireless_sleep_eeg(RAW_DATA_FILE, WIRELESS_FILTER_FILE);
end

if RUN_OVERNIGHT_VT_STFT
    plot_overnight_vt_stft(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_VT_DIR, OUT_STFT_DIR);
end

if RUN_DOCTOR_CHECK_30S_EXPORT
    export_doctor_check_30s_figures(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_DOCTOR_DIR);
end

if RUN_REPRESENTATIVE_30S_WAVEFORM
    export_representative_30s_waveform(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_VT_DIR);
end

if RUN_WHOLE_NIGHT_STACKED_WAVEFORM
    export_whole_night_stacked_waveform(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_VT_DIR);
end

if RUN_SLEEP_STAGE_AGREEMENT
    sleep_stage_agreement_numeric_local(SLEEP_STAGE_FILE, OUT_STAGE_DIR);
end

fprintf('\nAll selected PSG validation sleep-analysis steps finished.\n');

end


%% Wireless sleep EEG preprocessing

function preprocess_wireless_sleep_eeg(RAW_DATA_FILE, WIRELESS_FILTER_FILE)

fprintf('\n=== Preprocessing wireless sleep EEG ===\n');

load(RAW_DATA_FILE, 'raw_data_list');

timeFormat = 'dd-HH:MM';
data_set_num = length(raw_data_list);

eeg_data_flt_cell = cell(1, data_set_num);

for data_set_index = 1:data_set_num

    fs = raw_data_list(data_set_index).fs;

    % Original sleep-data filtering:
    % high-pass around 0.32 Hz and low-pass around 35 Hz.
    [eeg_data_temp, d] = highpass( ...
        double(raw_data_list(data_set_index).eeg_data_raw), ...
        0.32, fs, ...
        'ImpulseResponse', 'iir', ...
        'Steepness', 0.5); %#ok<ASGLU>

    f_flt = [0 33 40 fs/2] ./ (fs/2);
    m_flt = [1 1  0  0];

    b_flt = fir2(fix(fs * 10), f_flt, m_flt);

    eeg_data_flt_cell{data_set_index} = single(filtfilt(b_flt, 1, eeg_data_temp));
end

data_invalid_index_cell = cell(1, data_set_num);

for data_set_index = 1:data_set_num

    fs = raw_data_list(data_set_index).fs;
    data_loss_index = raw_data_list(data_set_index).data_loss_index;
    [data_num, ~] = size(eeg_data_flt_cell{data_set_index});

    filter_invalid_len_t = 20;
    data_invalid_index = int16(data_loss_index * 1);
    point_num_invalid_single_side = ceil(ceil(fs * filter_invalid_len_t) / 2);

    data_invalid_index(1:point_num_invalid_single_side) = 2;
    data_invalid_index((end - point_num_invalid_single_side + 1):end) = 2;

    for i = (point_num_invalid_single_side + 1):(data_num - point_num_invalid_single_side)

        if ~data_loss_index(i - 1) && data_loss_index(i)
            data_invalid_index((i - point_num_invalid_single_side):(i + point_num_invalid_single_side - 1)) = 2;

        elseif data_loss_index(i - 1) && ~data_loss_index(i)
            data_invalid_index((i - point_num_invalid_single_side):(i + point_num_invalid_single_side - 1)) = 2;
        end
    end

    data_invalid_index(data_loss_index) = 1;

    for i = 1:data_num
        if data_invalid_index(i) ~= 0
            eeg_data_flt_cell{data_set_index}(i, :) = 0;
        end
    end

    data_invalid_index_cell{data_set_index} = data_invalid_index;
end

data_invalid_array_cell = cell(1, data_set_num);

for data_set_index = 1:data_set_num

    [~, ch_num] = size(eeg_data_flt_cell{data_set_index});
    data_invalid_index = data_invalid_index_cell{data_set_index};

    % 0: good data
    % 1: communication data loss or electrode check
    % 2: filter-caused invalid data
    % 3: motion artifact
    % -1: electrode wire disconnect
    data_invalid_array_cell{data_set_index} = repmat(data_invalid_index, 1, ch_num);
end

eeg_data_flt = [];
timePoints = [];
t_list = [];
data_invalid_array = [];

sample_period = raw_data_list(1).sample_period;
fs = raw_data_list(1).fs;
startTime = raw_data_list(1).startTime;

for data_set_index = 1:data_set_num
    eeg_data_flt = cat(1, eeg_data_flt, eeg_data_flt_cell{data_set_index});
    timePoints = cat(1, timePoints, raw_data_list(data_set_index).timePoints);
    t_list = cat(1, t_list, raw_data_list(data_set_index).t_list);
    data_invalid_array = cat(1, data_invalid_array, data_invalid_array_cell{data_set_index});
end

save(WIRELESS_FILTER_FILE, ...
    'eeg_data_flt', 'timePoints', 'timeFormat', 't_list', ...
    'sample_period', 'fs', 'startTime', 'data_invalid_array', '-v7.3');

fprintf('Saved filtered wireless EEG to:\n%s\n', WIRELESS_FILTER_FILE);

end


%% Overnight wireless-vs-hospital trace and STFT comparison

function plot_overnight_vt_stft(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_VT_DIR, OUT_STFT_DIR)
% Generate whole-night raw EEG traces and STFT spectrograms for PSG validation.

fprintf('\n=== Whole-night raw traces and wireless-vs-hospital STFT comparison ===\n');

% Export separate whole-night raw signal figures first.
% These replace the previous overlaid wireless/hospital long trace.
export_whole_night_stacked_waveform(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_VT_DIR);

%% Load data for STFT spectrogram comparison

load(WIRELESS_FILTER_FILE, 'eeg_data_flt', 'fs', 'timePoints', 'data_invalid_array');

eeg_data = eeg_data_flt;

eeg_data_mask_zero = eeg_data;
eeg_data_mask_zero(data_invalid_array ~= 0) = 0;

hospital_data = load(HOSPITAL_EEG_FILE);

eeg_data_hospital = hospital_data.eeg_data_hospital ./ 1000; % mV
t_list_hospital = hospital_data.t_list;
fs_hospital = hospital_data.fs;

startTime_hospital = datetime(2024, 9, 24, 20, 34, 03);
timePoints_hospital = startTime_hospital + seconds(t_list_hospital);

%% Overnight STFT spectrogram comparison

plot_start_time_hospital = datetime(2024, 9, 24, 22, 30, 33);
plot_duration_s = 3600 * 8.5;
plot_channel_list = [1 2 3 4 5 6];
wireless_time_mismatch_s = 0;

time_tick_period = 60 * 60;
fig_size = [0, 0, 1100, 250];

plot_start_time = plot_start_time_hospital + seconds(wireless_time_mismatch_s);
plot_end_time = plot_start_time + seconds(plot_duration_s);
plot_end_time_hospital = plot_start_time_hospital + seconds(plot_duration_s);

[plot_start_index, plot_end_index] = timepoints2index(plot_start_time, plot_end_time, timePoints);
plot_index = plot_start_index:plot_end_index;

[plot_start_index_hospital, plot_end_index_hospital] = timepoints2index(plot_start_time_hospital, plot_end_time_hospital, timePoints_hospital);
plot_index_hospital = plot_start_index_hospital:plot_end_index_hospital;

plot_duration_h = hours(plot_end_time - plot_start_time);
start_time_time_str = datestr(plot_start_time - days(7), "HH-MM-SS");
data_title = sprintf("s-t channel %s @%s %.1fh@", num2str(plot_channel_list), start_time_time_str, plot_duration_h);

[T, timePoints_T, F, magnitude_spectrogram] = eeg_stft(eeg_data_mask_zero(plot_index, :) .* 1000, plot_start_time, 4, fs, 0.5);
power_spectrogram = mean(magnitude_spectrogram(:, :, plot_channel_list) .^ 2, 3);

[T_hospital, timePoints_T_hospital, F_hospital, magnitude_spectrogram_hospital] = eeg_stft(eeg_data_hospital(plot_index_hospital, :) .* 1000, plot_start_time_hospital, 4, fs_hospital, 0.5);
power_spectrogram_hospital = mean(magnitude_spectrogram_hospital(:, :, plot_channel_list) .^ 2, 3);

font_size = 20;
f_range = [0.5 35];

if exist('cbrewer2', 'file')
    color_map = cbrewer2('div', 'RdBu', 256);
    color_map = flipud(color_map);
else
    color_map = flipud(parula(256));
end

% dB-power spectrograms used for the sleep-validation figure.
power_db = true;
y_log = true;
color_range = [-20 20];
YTick = [1 3 10 30];

f_log_spec_dB = eeg_spectrogram_plot(timePoints_T, F, power_spectrogram, plot_start_time, plot_end_time, 1, f_range, time_tick_period, power_db, color_range, color_map, 'HH:MM', YTick, y_log, fig_size, sprintf('%s wireless', data_title), font_size);
f_log_spec_dB_hospital = eeg_spectrogram_plot(timePoints_T_hospital, F_hospital, power_spectrogram_hospital, plot_start_time_hospital, plot_end_time_hospital, 1, f_range, time_tick_period, power_db, color_range, color_map, 'HH:MM', YTick, y_log, fig_size, sprintf('%s hospital', data_title), font_size);

y_log = false;
YTick = [1 10 20 30];

f_linear_spec_dB = eeg_spectrogram_plot(timePoints_T, F, power_spectrogram, plot_start_time, plot_end_time, 1, f_range, time_tick_period, power_db, color_range, color_map, 'HH:MM', YTick, y_log, fig_size, sprintf('%s wireless', data_title), font_size);
f_linear_spec_dB_hospital = eeg_spectrogram_plot(timePoints_T_hospital, F_hospital, power_spectrogram_hospital, plot_start_time_hospital, plot_end_time_hospital, 1, f_range, time_tick_period, power_db, color_range, color_map, 'HH:MM', YTick, y_log, fig_size, sprintf('%s hospital', data_title), font_size);

exportgraphics(f_log_spec_dB, fullfile(OUT_STFT_DIR, sprintf('%s wireless log_freq dB_power.png', data_title)), 'Resolution', 300);
exportgraphics(f_linear_spec_dB, fullfile(OUT_STFT_DIR, sprintf('%s wireless linear_freq dB_power.png', data_title)), 'Resolution', 300);

exportgraphics(f_log_spec_dB_hospital, fullfile(OUT_STFT_DIR, sprintf('%s hospital log_freq dB_power.png', data_title)), 'Resolution', 300);
exportgraphics(f_linear_spec_dB_hospital, fullfile(OUT_STFT_DIR, sprintf('%s hospital linear_freq dB_power.png', data_title)), 'Resolution', 300);

close(f_log_spec_dB);
close(f_linear_spec_dB);
close(f_log_spec_dB_hospital);
close(f_linear_spec_dB_hospital);

% STFT source data.
writetable(table([NaN; T], [F'; 10 * log10(power_spectrogram')], ...
    'VariableNames', ["time (s)", "frequency (Hz); power spectrum (dB)"]), ...
    fullfile(OUT_STFT_DIR, sprintf('%s wireless dB_power.csv', data_title)));

writetable(table([NaN; T_hospital], [F_hospital'; 10 * log10(power_spectrogram_hospital')], ...
    'VariableNames', ["time (s)", "frequency (Hz); power spectrum (dB)"]), ...
    fullfile(OUT_STFT_DIR, sprintf('%s hospital dB_power.csv', data_title)));

fprintf('Saved whole-night raw traces and STFT outputs. Overlaid long trace and PSD-comparison outputs were intentionally omitted.\n');

end


%% Doctor-check 30 s export

function export_doctor_check_30s_figures(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_DOCTOR_DIR)

fprintf('\n=== Exporting 30 s doctor-check figures ===\n');

load(WIRELESS_FILTER_FILE, 'eeg_data_flt', 'fs', 'timePoints', 'data_invalid_array');

eeg_data = eeg_data_flt;
eeg_data_mask = eeg_data;
eeg_data_mask(data_invalid_array ~= 0) = NaN;

electrode_name_str_list = ["F3-A1"; "F4-A1"; "C3-A1"; "C4-A1"; "O1-A1"; "O2-A1"];

hospital_data = load(HOSPITAL_EEG_FILE);
eeg_data_hospital = hospital_data.eeg_data_hospital ./ 1000;
t_list_hospital = hospital_data.t_list;
fs_hospital = hospital_data.fs;
startTime_hospital = datetime(2024, 9, 24, 20, 34, 03);
timePoints_hospital = startTime_hospital + seconds(t_list_hospital);

electrode_name_hospital_str_list = ["F3-M2"; "F4-M1"; "C3-M2"; "C4-M1"; "O1-M2"; "O2-M1"];

ch_split_uv = 100;
scale_line_range_1 = 35;
scale_line_range_2 = 75;
plot_period = 30;
time_tick_period = 2;
edge_uv_down = 50;
edge_uv_up = 10;
scale_line_color = [0.7 0.7 0.9];
x_edge_s = 0.5;
font_size = 20;
fig_size = [0, 100, 2000, 1200];

combined_out = fullfile(OUT_DOCTOR_DIR, 'wireless_hospital_combined');
wireless_out = fullfile(OUT_DOCTOR_DIR, 'wireless_only');
hospital_out = fullfile(OUT_DOCTOR_DIR, 'hospital_only');

ensure_dir(combined_out);
ensure_dir(wireless_out);
ensure_dir(hospital_out);

% Combined wireless + hospital figures.
for figure_index = 1:1030

    plot_start_time = datetime(2024, 9, 24, 22, 25, 03) + seconds((figure_index - 1) * 30);

    f = figure('Visible', 'off');

    subplot(2, 1, 1);
    eeg_sleep_check_plot_local(plot_start_time, figure_index, ...
        ch_split_uv, scale_line_range_1, scale_line_range_2, plot_period, time_tick_period, ...
        edge_uv_down, edge_uv_up, scale_line_color, x_edge_s, font_size, fig_size, ...
        electrode_name_str_list, timePoints, eeg_data_mask, fs);
    title('Wireless EEG');

    subplot(2, 1, 2);
    eeg_sleep_check_plot_local(plot_start_time, figure_index, ...
        ch_split_uv, scale_line_range_1, scale_line_range_2, plot_period, time_tick_period, ...
        edge_uv_down, edge_uv_up, scale_line_color, x_edge_s, font_size, fig_size, ...
        electrode_name_hospital_str_list, timePoints_hospital, eeg_data_hospital, fs_hospital);
    title('Hospital PSG EEG');

    exportgraphics(f, fullfile(combined_out, sprintf('%d.png', figure_index)), 'Resolution', 200);
    close(f);
end

fprintf('Saved 30 s doctor-check combined figures to:\n%s\n', combined_out);

end


%% Representative 30 s stacked waveform

function export_representative_30s_waveform(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_VT_DIR)

fprintf('\n=== Exporting representative 30 s stacked waveforms ===\n');

load(WIRELESS_FILTER_FILE, 'eeg_data_flt', 'fs', 'timePoints', 'data_invalid_array');

eeg_data = eeg_data_flt;
fs_wireless = fs;
timePoints_wireless = timePoints;

h = load(HOSPITAL_EEG_FILE, 'eeg_data_hospital', 'fs', 't_list');
eeg_data_hospital = h.eeg_data_hospital / 1000;
fs_hospital = h.fs;
t_list_hospital = h.t_list;
startTime_hospital = datetime(2024, 9, 24, 20, 34, 03);
timePoints_hospital = startTime_hospital + seconds(t_list_hospital);

chan_names = ["F3", "F4", "C3", "C4", "O1", "O2"];
ch_num = numel(chan_names);

wireless_col = [59 136 167] ./ 256;
hospital_col = [40 54 89] ./ 256;

eeg_data_mask_zero = eeg_data;
eeg_data_mask_zero(data_invalid_array ~= 0) = 0;

plot_start_time_hospital = datetime(2024, 9, 25, 02, 02, 03);
plot_duration_s = 30;
wireless_time_mismatch_s = -0.11;

plot_start_time_wireless = plot_start_time_hospital + seconds(wireless_time_mismatch_s);
plot_end_time_hospital = plot_start_time_hospital + seconds(plot_duration_s);
plot_end_time_wireless = plot_start_time_wireless + seconds(plot_duration_s);

[si_w, ei_w] = timepoints2index(plot_start_time_wireless, plot_end_time_wireless, timePoints_wireless);
plot_index_w = si_w:ei_w;

[si_h, ei_h] = timepoints2index(plot_start_time_hospital, plot_end_time_hospital, timePoints_hospital);
plot_index_h = si_h:ei_h;

t_plot_w = (0:length(plot_index_w)-1) ./ fs_wireless;
t_plot_h = (0:length(plot_index_h)-1) ./ fs_hospital;

offset = 100;
y_offsets = offset * (ch_num-1:-1:0);

fig = figure('Visible', 'off', 'Position', [0 100 600 300]);
hold on;
for ch = 1:ch_num
    y = eeg_data_hospital(plot_index_h, ch) * 1000 + y_offsets(ch);
    idx_nan = (eeg_data_hospital(plot_index_h, ch) == 0 | isnan(eeg_data_hospital(plot_index_h, ch)));
    y(idx_nan) = NaN;
    plot(t_plot_h, y, 'Color', hospital_col, 'LineWidth', 0.1);
end
yticks_vals = sort(y_offsets);
yticklabels_local = flip(chan_names);
set(gca, 'YDir', 'reverse', 'YTick', yticks_vals, 'YTickLabel', yticklabels_local, ...
    'FontSize', 16, 'Box', 'off');
xlabel('Time (s)', 'FontSize', 18);
title('Hospital 6-channel stacked waveform (30 s)', 'FontSize', 18, 'FontWeight', 'bold');
set(gcf, 'Color', 'w');
exportgraphics(fig, fullfile(OUT_VT_DIR, 'hospital_waveform_6ch_stacked_s.png'), 'Resolution', 300);
close(fig);

fig2 = figure('Visible', 'off', 'Position', [0 100 600 300]);
hold on;
for ch = 1:ch_num
    y = eeg_data_mask_zero(plot_index_w, ch) * 1000 + y_offsets(ch);
    idx_nan = (eeg_data_mask_zero(plot_index_w, ch) == 0 | isnan(eeg_data_mask_zero(plot_index_w, ch)));
    y(idx_nan) = NaN;
    plot(t_plot_w, y, 'Color', wireless_col, 'LineWidth', 0.1);
end
set(gca, 'YDir', 'reverse', 'YTick', yticks_vals, 'YTickLabel', yticklabels_local, ...
    'FontSize', 16, 'Box', 'off');
xlabel('Time (s)', 'FontSize', 18);
title('Wireless 6-channel stacked waveform (30 s)', 'FontSize', 18, 'FontWeight', 'bold');
set(gcf, 'Color', 'w');
exportgraphics(fig2, fullfile(OUT_VT_DIR, 'wireless_waveform_6ch_stacked_s.png'), 'Resolution', 300);
close(fig2);

T_h = array2table([t_plot_h' eeg_data_hospital(plot_index_h, :) * 1000], ...
    'VariableNames', ["Time_s", strcat("Hospital_", chan_names, "_uV")]);
writetable(T_h, fullfile(OUT_VT_DIR, 'hospital_waveform_6ch_s.csv'));

T_w = array2table([t_plot_w' eeg_data_mask_zero(plot_index_w, :) * 1000], ...
    'VariableNames', ["Time_s", strcat("Wireless_", chan_names, "_uV")]);
writetable(T_w, fullfile(OUT_VT_DIR, 'wireless_waveform_6ch_s.csv'));

fprintf('Saved representative 30 s waveforms.\n');

end


%% Whole-night stacked waveform

function export_whole_night_stacked_waveform(WIRELESS_FILTER_FILE, HOSPITAL_EEG_FILE, OUT_VT_DIR)

fprintf('\n=== Exporting whole-night stacked waveforms ===\n');

load(WIRELESS_FILTER_FILE, 'eeg_data_flt', 'fs', 'timePoints', 'data_invalid_array');

eeg_data = eeg_data_flt;
fs_wireless = fs;
timePoints_wireless = timePoints;

h = load(HOSPITAL_EEG_FILE, 'eeg_data_hospital', 'fs', 't_list');
eeg_data_hospital = h.eeg_data_hospital / 1000;
fs_hospital = h.fs;
t_list_hospital = h.t_list;
startTime_hospital = datetime(2024, 9, 24, 20, 34, 03);
timePoints_hospital = startTime_hospital + seconds(t_list_hospital);

chan_names = ["F3", "F4", "C3", "C4", "O1", "O2"];
ch_num = 6;

wireless_col = [59 136 167] ./ 256;
hospital_col = [40 54 89] ./ 256;

eeg_data_mask_zero = eeg_data;
eeg_data_mask_zero(data_invalid_array ~= 0) = 0;

plot_start_time_hospital = datetime(2024, 9, 24, 22, 30, 33);
plot_duration_s = 3600 * 8.5;
wireless_time_mismatch_s = -0.11;

plot_start_time_wireless = plot_start_time_hospital + seconds(wireless_time_mismatch_s);
plot_end_time_hospital = plot_start_time_hospital + seconds(plot_duration_s);
plot_end_time_wireless = plot_start_time_wireless + seconds(plot_duration_s);

[si_w, ei_w] = timepoints2index(plot_start_time_wireless, plot_end_time_wireless, timePoints_wireless);
plot_index_w = si_w:ei_w;

[si_h, ei_h] = timepoints2index(plot_start_time_hospital, plot_end_time_hospital, timePoints_hospital);
plot_index_h = si_h:ei_h;

t_plot_w = (0:length(plot_index_w)-1) ./ fs_wireless / 3600;
t_plot_h = (0:length(plot_index_h)-1) ./ fs_hospital / 3600;

offset = 400;
y_offsets = offset * (ch_num-1:-1:0);

fig = figure('Visible', 'off', 'Position', [0 100 3000 300]);
hold on;
for ch = 1:ch_num
    y = eeg_data_hospital(plot_index_h, ch) * 1000 + y_offsets(ch);
    idx_nan = (eeg_data_hospital(plot_index_h, ch) == 0 | isnan(eeg_data_hospital(plot_index_h, ch)));
    y(idx_nan) = NaN;
    plot(t_plot_h, y, 'Color', hospital_col, 'LineWidth', 0.7);
end
set(gca, 'YTick', fliplr(y_offsets), 'YTickLabel', fliplr(chan_names), ...
    'FontSize', 16, 'Box', 'off', 'YGrid', 'off', 'XGrid', 'off', 'TickDir', 'out');
ylim([y_offsets(end)-offset/2, y_offsets(1)+offset/2]);
xlim([t_plot_h(1), t_plot_h(end)]);
xlabel('Time (h)', 'FontSize', 18);
title('Hospital 6-channel stacked waveform', 'FontWeight', 'bold', 'FontSize', 18);
set(gcf, 'Color', 'w');
exportgraphics(fig, fullfile(OUT_VT_DIR, 'hospital_waveform_6ch_stacked_h.png'), 'Resolution', 300);
close(fig);

fig2 = figure('Visible', 'off', 'Position', [0 100 3000 300]);
hold on;
for ch = 1:ch_num
    y = eeg_data_mask_zero(plot_index_w, ch) * 1000 + y_offsets(ch);
    idx_nan = (eeg_data_mask_zero(plot_index_w, ch) == 0 | isnan(eeg_data_mask_zero(plot_index_w, ch)));
    y(idx_nan) = NaN;
    plot(t_plot_w, y, 'Color', wireless_col, 'LineWidth', 0.7);
end
set(gca, 'YTick', fliplr(y_offsets), 'YTickLabel', fliplr(chan_names), ...
    'FontSize', 16, 'Box', 'off', 'YGrid', 'off', 'XGrid', 'off', 'TickDir', 'out');
ylim([y_offsets(end)-offset/2, y_offsets(1)+offset/2]);
xlim([t_plot_w(1), t_plot_w(end)]);
xlabel('Time (h)', 'FontSize', 18);
title('Wireless 6-channel stacked waveform', 'FontWeight', 'bold', 'FontSize', 18);
set(gcf, 'Color', 'w');
exportgraphics(fig2, fullfile(OUT_VT_DIR, 'wireless_waveform_6ch_stacked_h.png'), 'Resolution', 300);
close(fig2);

tbl_h = array2table([t_plot_h' eeg_data_hospital(plot_index_h, :) * 1000], ...
    'VariableNames', ["Time_h", strcat("Hospital_", chan_names, "_uV")]);
writetable(tbl_h, fullfile(OUT_VT_DIR, 'hospital_waveform_6ch_h.csv'));

tbl_w = array2table([t_plot_w' eeg_data_mask_zero(plot_index_w, :) * 1000], ...
    'VariableNames', ["Time_h", strcat("Wireless_", chan_names, "_uV")]);
writetable(tbl_w, fullfile(OUT_VT_DIR, 'wireless_waveform_6ch_h.csv'));

fprintf('Saved whole-night stacked waveforms.\n');

end


%% Sleep-stage agreement

function sleep_stage_agreement_numeric_local(inputFile, outdir)

fprintf('\n=== Sleep-stage agreement and Cohen kappa ===\n');

if ~isfile(inputFile)
    warning('Sleep-stage file was not found:\n%s', inputFile);
    return;
end

if ~exist(outdir, 'dir')
    mkdir(outdir);
end

T = readtable(inputFile, 'VariableNamingRule', 'preserve');
assert(width(T) >= 3, 'Input file should contain at least three columns: time, hospital, wireless.');

hospital = T{:, 2};
wireless = T{:, 3};

valid = ismember(hospital, 1:5) & ismember(wireless, 1:5);
hospital = hospital(valid);
wireless = wireless(valid);

order_codes = [5 4 3 2 1]; % W,N1,N2,N3,R
stage_labels = {'W', 'N1', 'N2', 'N3', 'R'};

map = zeros(1, 5);
for k = 1:5
    map(order_codes(k)) = k;
end

hi = arrayfun(@(x) map(x), hospital);
wi = arrayfun(@(x) map(x), wireless);

C = accumarray([hi, wi], 1, [5, 5], @sum, 0);

row_sum = sum(C, 2);
P_row = C ./ max(row_sum, 1) * 100;

N = sum(row_sum);
Po = sum(diag(C)) / N;
col_sum = sum(C, 1)';
Pe = sum((row_sum .* col_sum)) / (N^2);
kappa = (Po - Pe) / (1 - Pe);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 720 640]);
imagesc(P_row, [0 100]);
axis image;
colormap(flipud(hot));
colorbar;

set(gca, 'XTick', 1:5, 'XTickLabel', stage_labels, ...
         'YTick', 1:5, 'YTickLabel', stage_labels, ...
         'XAxisLocation', 'top', 'FontSize', 11);

xlabel('Scoring with wireless patch');
ylabel('Scoring with PSG (hospital)');
title(sprintf('%.2f%% agreement, \\kappa = %.2f', Po * 100, kappa), 'FontSize', 12);

for r = 1:5
    for c = 1:5
        pct = P_row(r, c);
        cnt = C(r, c);

        txtColor = "k";
        if pct > 60
            txtColor = "w";
        end

        text(c, r, sprintf('%.2f%%\n(%d)', pct, cnt), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 10, ...
            'FontWeight', 'bold', ...
            'Color', txtColor);
    end
end

set(gca, 'TickLength', [0 0]);
exportgraphics(fig, fullfile(outdir, 'confusion_heatmap.png'), 'Resolution', 300);
close(fig);

C_tbl = array2table(C, 'VariableNames', stage_labels, 'RowNames', stage_labels);
writetable(addRowNameCol(C_tbl), fullfile(outdir, 'confusion_counts.csv'));

P_tbl = array2table(P_row, 'VariableNames', stage_labels, 'RowNames', stage_labels);
writetable(addRowNameCol(P_tbl), fullfile(outdir, 'confusion_percent_row.csv'));

metrics = table(N, Po, Pe, kappa, ...
    'VariableNames', {'N', 'Agreement_Po', 'Expected_Pe', 'CohenKappa'});
writetable(metrics, fullfile(outdir, 'metrics.csv'));

fprintf('Saved sleep-stage agreement results to:\n%s\n', outdir);

end


%% ========================================================================
%  Utility functions
%  ========================================================================

function ensure_dir(folder_path)
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
end
end


function [start_index, end_index] = timepoints2index(start_time, end_time, timePoints)

start_index = find(timePoints >= start_time, 1);

if isempty(start_index)
    error('Start time is later than all time points.');
end

end_index = find(timePoints > end_time, 1) - 1;

if isempty(end_index)
    end_index = length(timePoints);
end

if end_index < 1
    error('End time is earlier than all time points.');
end

end


function [T, timePoints_T, F, magnitude_spectrogram] = eeg_stft(eeg_data, startTime, window_period, sample_rate, overlap_rate)

window_length = ceil(window_period * sample_rate);

if window_length > size(eeg_data, 1)
    window_length = size(eeg_data, 1);
end

overlap = round(overlap_rate * window_length);
window_function = hann(window_length);

[S, F, T] = stft(eeg_data, sample_rate, ...
    "Window", window_function, ...
    "OverlapLength", overlap);

p_spectrum = abs(S).^2;

% Single-sided spectrum handling, following the original workflow.
p_spectrum = p_spectrum(ceil(window_length/2):end, :, :);
p_spectrum(2:ceil(window_length/2), :, :) = p_spectrum(2:ceil(window_length/2), :, :) .* 2;
p_spectrum = p_spectrum ./ (sample_rate * window_length);

magnitude_spectrogram = sqrt(p_spectrum);

F = F(ceil(window_length/2):end);
timePoints_T = startTime + seconds(T);

end


function figure_spectrogram = eeg_spectrogram_plot(timePoints_T, F, power_spectrogram, ...
    plot_start_time, plot_end_time, plot_ch_index, f_range, time_tick_period, ...
    power_db, color_range, color_map, timeFormat, y_tick, y_log, fig_size, data_title, font_size)

if ndims(power_spectrogram) == 2
    power_spectrogram = reshape(power_spectrogram, size(power_spectrogram, 1), size(power_spectrogram, 2), 1);
end

[plot_start_index, plot_end_index] = timepoints2index(plot_start_time, plot_end_time, timePoints_T);

T = seconds(timePoints_T - timePoints_T(1));
startTime = timePoints_T(1);

time_ticks = seconds(plot_start_time - startTime):time_tick_period:seconds(plot_end_time - startTime);
timePoints_ticks = startTime + seconds(time_ticks);
timeLabels_ticks = datestr(timePoints_ticks, timeFormat);

figure_spectrogram = figure('Visible', 'off');
sgtitle(data_title, 'FontSize', 12, 'Interpreter', 'none');
set(gcf, 'Position', fig_size);

for i = plot_ch_index

    subplot(length(plot_ch_index), 1, i - plot_ch_index(1) + 1);

    T_plot_range = T(plot_start_index:plot_end_index);
    F_mask = (F >= f_range(1)) & (F <= f_range(2));
    F_plot_range = F(F_mask);

    if power_db
        Power_plot = 10 * log10(power_spectrogram(F_mask, plot_start_index:plot_end_index, i));
    else
        Power_plot = power_spectrogram(F_mask, plot_start_index:plot_end_index, i);
    end

    if y_log
        F_plot_range_log = log10(F_plot_range);
        F_plot_interp = F_plot_range_log(1):0.001:F_plot_range_log(end);

        Power_plot = interp2(double(T_plot_range), double(F_plot_range_log), Power_plot, ...
                             double(T_plot_range), double(F_plot_interp));

        F_plot_range = F_plot_interp;
    end

    imagesc([T_plot_range(1), T_plot_range(end)], ...
            [F_plot_range(1), F_plot_range(end)], ...
            Power_plot);

    colormap(color_map);
    axis xy;

    ax = gca;
    ax.XTick = time_ticks;
    ax.XTickLabel = timeLabels_ticks;

    xlim(seconds([plot_start_time, plot_end_time] - startTime));
    ylabel('Frequency (Hz)');
    ylim([F_plot_range(1), F_plot_range(end)]);

    if y_log
        ax.YTick = log10(y_tick);
        ax.YTickLabel = arrayfun(@num2str, y_tick, 'UniformOutput', false);
    else
        ax.YTick = y_tick;
    end

    c = colorbar;

    if power_db
        c.Label.String = 'Power (dB)';
    else
        c.Label.String = 'Power (uV^2/Hz)';
    end

    caxis(color_range);
    set(gca, 'FontSize', font_size);
end

xlabel('Time');

end


function eeg_sleep_check_plot_local(plot_start_time, figure_index, ...
    ch_split_uv, scale_line_range_1, scale_line_range_2, plot_period, time_tick_period, ...
    edge_uv_down, edge_uv_up, scale_line_color, x_edge_s, font_size, fig_size, ...
    electrode_name_str_list, timePoints, eeg_data, fs)

[~, ch_num] = size(eeg_data);

plot_timePoints = 0;
plot_end_time = plot_start_time + seconds(plot_period);

start_time_time_str = datestr(plot_start_time, "HH:MM:SS");
data_title = sprintf("Epoch index: %d, start time: %s", figure_index, start_time_time_str);

[plot_start_index, plot_end_index] = timepoints2index(plot_start_time, plot_end_time, timePoints);

plot_decimation = 1;
plot_index = plot_start_index:plot_decimation:plot_end_index;

time_ticks_s = 0:time_tick_period:plot_period;

if plot_timePoints == 1
    t_plot = timePoints(plot_index);
    time_ticks = seconds(time_ticks_s) + plot_start_time;
else
    t_plot = (0:(length(plot_index)-1)) ./ fs;
    time_ticks = time_ticks_s;
end

set(gcf, 'Position', fig_size);
hold on;

y_bias = zeros(ch_num, 1);

for ch_index = 1:ch_num

    y_bias(ch_index) = ch_split_uv * (ch_num - ch_index);

    scale_line_x = [t_plot(1), t_plot(end)];
    scale_line_y_1_up = [scale_line_range_1 / 2, scale_line_range_1 / 2] + y_bias(ch_index);
    scale_line_y_1_down = [-scale_line_range_1 / 2, -scale_line_range_1 / 2] + y_bias(ch_index);
    scale_line_y_2_up = [scale_line_range_2 / 2, scale_line_range_2 / 2] + y_bias(ch_index);
    scale_line_y_2_down = [-scale_line_range_2 / 2, -scale_line_range_2 / 2] + y_bias(ch_index);

    plot(scale_line_x, scale_line_y_1_up, '--', 'Color', scale_line_color, 'LineWidth', 0.75);
    plot(scale_line_x, scale_line_y_1_down, '--', 'Color', scale_line_color, 'LineWidth', 0.75);
    plot(scale_line_x, scale_line_y_2_up, '-', 'Color', scale_line_color, 'LineWidth', 1);
    plot(scale_line_x, scale_line_y_2_down, '-', 'Color', scale_line_color, 'LineWidth', 1);
end

if plot_timePoints == 1
    scale_line_x_1 = [t_plot(1), t_plot(1) + seconds(1)];
    scale_line_x_2 = [t_plot(1) + seconds(2), t_plot(1) + seconds(3)];
else
    scale_line_x_1 = [0, 1];
    scale_line_x_2 = [2, 3];
end

y_bias_indication = ch_split_uv * ch_num;

plot(scale_line_x_2, [scale_line_range_1 / 2, scale_line_range_1 / 2] + y_bias_indication, '--', 'Color', scale_line_color, 'LineWidth', 0.75);
plot(scale_line_x_2, [-scale_line_range_1 / 2, -scale_line_range_1 / 2] + y_bias_indication, '--', 'Color', scale_line_color, 'LineWidth', 0.75);
plot(scale_line_x_1, [scale_line_range_2 / 2, scale_line_range_2 / 2] + y_bias_indication, '-', 'Color', scale_line_color, 'LineWidth', 1);
plot(scale_line_x_1, [-scale_line_range_2 / 2, -scale_line_range_2 / 2] + y_bias_indication, '-', 'Color', scale_line_color, 'LineWidth', 1);

text(mean(scale_line_x_1), y_bias_indication, '75uV', 'HorizontalAlignment', 'center', 'FontSize', 14);
text(mean(scale_line_x_2), y_bias_indication, '35uV', 'HorizontalAlignment', 'center', 'FontSize', 14);
text((t_plot(1) + t_plot(end)) / 2, y_bias_indication + 20, data_title, ...
    'HorizontalAlignment', 'center', 'FontSize', font_size + 4);

for ch_index = 1:ch_num
    ch_data = eeg_data(plot_index, ch_index) .* 1000;
    plot(t_plot, ch_data + y_bias(ch_index), 'k', 'LineWidth', 0.5);
end

ax = gca;
ax.XTick = time_ticks;
ax.YTick = y_bias(end:-1:1);
ax.YTickLabel = electrode_name_str_list(end:-1:1);

ylim([-scale_line_range_2 / 2 - edge_uv_down, y_bias_indication + scale_line_range_2 / 2 + edge_uv_up]);

if plot_timePoints == 1
    xlim([plot_start_time - seconds(x_edge_s), plot_end_time + seconds(x_edge_s)]);
else
    xlim([0 - x_edge_s, plot_period + x_edge_s]);
end

xlabel('t (s)');
set(gca, 'FontSize', font_size);

end


function T2 = addRowNameCol(T)

rn = string(T.Properties.RowNames);
colName = 'Stage';

if any(strcmp(T.Properties.VariableNames, colName))
    colName = [colName '_1'];
end

T2 = addvars(T, rn, 'Before', 1, 'NewVariableNames', colName);
T2.Properties.RowNames = {};

end
