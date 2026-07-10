clear; clc; close all force;

%% Configuration
RAW_DATA_FILE = fullfile('example_data', 'raw_data.mat');
LABEL_FILE    = fullfile('example_data', 'label.mat');

OUTPUT_DIR  = fullfile('outputs', 'filtered_eeg');
OUTPUT_FILE = fullfile(OUTPUT_DIR, 'eeg_data_filtered.mat');

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

% Processing switches
APPLY_F3_DISCONNECTION_CORRECTION = true;

% Set true only when you want to save a filter-response figure.
% For GitHub/demo runs, false is more stable.
PLOT_FILTER_RESPONSE = false;

%% Load raw EEG data

if ~isfile(RAW_DATA_FILE)
    error(['RAW_DATA_FILE not found: %s\n', ...
           'Please place raw_data.mat in example_data/ or edit RAW_DATA_FILE.'], RAW_DATA_FILE);
end

load(RAW_DATA_FILE);

required_raw_vars = {'eeg_data_raw', 'timePoints', 't_list', ...
                     'sample_period', 'fs', 'startTime', 'data_loss_index'};

for k = 1:numel(required_raw_vars)
    if ~exist(required_raw_vars{k}, 'var')
        error('Required variable "%s" is missing from raw_data.mat.', required_raw_vars{k});
    end
end

if isfile(LABEL_FILE)
    load(LABEL_FILE);
else
    warning('LABEL_FILE not found: %s. Electrode-check periods will not be marked from activity labels.', LABEL_FILE);
    timePoints_arr = [];
end

[data_num, ch_num] = size(eeg_data_raw);
data_loss_index = logical(data_loss_index(:));

fprintf('Loaded raw EEG data.\n');
fprintf('Data points: %d\n', data_num);
fprintf('Channels: %d\n', ch_num);
fprintf('Sampling rate: %.6f Hz\n\n', fs);

%% Channel reordering

% Raw channel order from the acquisition file
pos_name_raw = {'pz','p3','c3','c4','cz','f4','f3','fz', ...
                'p7','t7','p8','t8','o1','oz','o2','p4'};

% Standard channel order used for downstream analysis and scalp topography
pos_name_std = {'o2','oz','o1','p8','p4','pz','p3','p7', ...
                't8','c4','cz','c3','t7','f4','fz','f3'};

if ch_num ~= numel(pos_name_raw)
    error('Expected %d channels, but eeg_data_raw has %d channels.', numel(pos_name_raw), ch_num);
end

ch_index_swap = zeros(1, ch_num);

for i = 1:ch_num
    idx = find(strcmpi(pos_name_std{i}, pos_name_raw), 1);

    if isempty(idx)
        error('Channel "%s" was not found in the raw channel list.', pos_name_std{i});
    end

    ch_index_swap(i) = idx;
end

fprintf('Channel reordering index:\n');
disp(ch_index_swap);

% Reorder raw EEG data to the standard montage order.
eeg_data_raw = eeg_data_raw(:, ch_index_swap);

%% FIR band-pass filtering

f_flt = [0 0.5 1.5 33 40 fs/2] ./ (fs/2);
m_flt = [0 0   1   1  0  0];

filter_order = fix(fs * 10);
b_flt = fir2(filter_order, f_flt, m_flt);

if PLOT_FILTER_RESPONSE
    f_filter = figure('Visible', 'off', 'Renderer', 'painters');
    freqz(b_flt, 1, [], fs);
    title('FIR band-pass filter response');
    exportgraphics(f_filter, fullfile(OUTPUT_DIR, 'FIR_filter_response.png'), 'Resolution', 300);
    close(f_filter);
end

fprintf('Applying zero-phase FIR filtering...\n');

% Zero-phase filtering compensates filter latency.
eeg_data_flt = single(filtfilt(b_flt, 1, double(eeg_data_raw)));

%% F3 disconnection handling

% During the original long-term recording, the F3 wire was disconnected.
% The original workflow approximated F3 using the average of neighboring
% channels C3 and Fz during this period.
%
% In the standard channel order:
%   ch12 = C3
%   ch15 = Fz
%   ch16 = F3

disconn_start_index = [];
disconn_fix_index = [];

if APPLY_F3_DISCONNECTION_CORRECTION

    disconn_start_time = datetime(2024, 5, 10, 1, 17, 8);
    disconn_fix_time   = datetime(2024, 5, 10, 11, 34, 56);

    [disconn_start_index, disconn_fix_index] = timepoints2index( ...
        disconn_start_time, disconn_fix_time, timePoints);

    eeg_data_flt(disconn_start_index:disconn_fix_index, 16) = ...
        (eeg_data_flt(disconn_start_index:disconn_fix_index, 12) + ...
         eeg_data_flt(disconn_start_index:disconn_fix_index, 15)) ./ 2;

    fprintf('F3 disconnection period handled from %s to %s.\n', ...
        string(disconn_start_time), string(disconn_fix_time));
end

%% Mark data loss and filter-edge invalid samples

% In the original activity label list, label 19 corresponds to checking electrodes.
if ~isempty(timePoints_arr)

    if numel(timePoints_arr) ~= data_num
        warning('timePoints_arr length does not match EEG data length. Electrode-check marking was skipped.');
    else
        data_loss_index(timePoints_arr == 19) = true;
    end
end

% data_invalid_index:
%   0 = valid data
%   1 = communication data loss or electrode check
%   2 = filter-edge invalid data
data_invalid_index = int16(data_loss_index);

% The FIR forward-backward filter affects samples near data-loss boundaries.
% The original workflow marked a 20 s window around each boundary as invalid.
filter_invalid_len_t = 20;  % seconds
point_num_invalid_single_side = ceil(ceil(fs * filter_invalid_len_t) / 2);

% Mark beginning and end of the full recording.
data_invalid_index(1:point_num_invalid_single_side) = 2;
data_invalid_index((end - point_num_invalid_single_side + 1):end) = 2;

% Mark samples around data-loss start/recovery transitions.
for i = (point_num_invalid_single_side + 1):(data_num - point_num_invalid_single_side)

    if ~data_loss_index(i-1) && data_loss_index(i)
        % Data-loss start
        data_invalid_index((i - point_num_invalid_single_side):(i + point_num_invalid_single_side - 1)) = 2;

    elseif data_loss_index(i-1) && ~data_loss_index(i)
        % Data-loss recovery
        data_invalid_index((i - point_num_invalid_single_side):(i + point_num_invalid_single_side - 1)) = 2;
    end
end

% Keep actual data-loss samples as code 1.
data_invalid_index(data_loss_index) = 1;

%% ======================= Zero invalid samples before artifact marking =======================

% This prevents already-invalid/lost periods from triggering false
% large-amplitude artifact labels.
eeg_data_flt(data_invalid_index ~= 0, :) = 0;

%% Motion artifact marking

% Original conservative artifact marking:
% If absolute EEG amplitude exceeds 0.100 mV = 100 uV, mark a 4 s window
% centered around that point as motion/artifact-contaminated.
motion_arti_th = 0.100;        % mV
motion_arti_invalid_len_t = 4; % seconds
point_num_invalid_single_side = ceil(ceil(fs * motion_arti_invalid_len_t) / 2);

data_invalid_array = repmat(data_invalid_index, 1, ch_num);

fprintf('Marking motion artifacts using ±100 uV threshold and 4 s window...\n');

for ch_index = 1:ch_num

    artifact_points = find(abs(eeg_data_flt(:, ch_index)) > motion_arti_th);

    for k = 1:numel(artifact_points)

        data_index = artifact_points(k);

        artifact_range = ...
            (data_index - point_num_invalid_single_side):(data_index + point_num_invalid_single_side - 1);

        artifact_range = artifact_range(artifact_range >= 1 & artifact_range <= data_num);

        for artifact_index = artifact_range
            if data_invalid_array(artifact_index, ch_index) == 0
                data_invalid_array(artifact_index, ch_index) = 3;
            end
        end
    end
end

%% Mark F3 electrode disconnection

if APPLY_F3_DISCONNECTION_CORRECTION && ~isempty(disconn_start_index)
    data_invalid_array(disconn_start_index:disconn_fix_index, 16) = -1;
end

%% Save filtered EEG data

timeFormat = 'dd-HH:MM';

save(OUTPUT_FILE, ...
    'eeg_data_flt', ...
    'timePoints', ...
    'timeFormat', ...
    't_list', ...
    'sample_period', ...
    'fs', ...
    'startTime', ...
    'data_invalid_array', ...
    'pos_name_std', ...
    '-v7.3');

fprintf('\nFiltered EEG data saved to:\n%s\n', OUTPUT_FILE);
fprintf('Preprocessing finished.\n');

%% Local function

function [start_index, end_index] = timepoints2index(start_time, end_time, timePoints)
%TIMEPOINTS2INDEX Convert start and end datetime values to sample indices.
%
% start_index: first sample with timePoints >= start_time
% end_index:   last sample with timePoints <= end_time

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
