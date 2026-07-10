function motion_artifact_fft_wtc_snr()
% This script combines:
%   1. FFT harmonic spectrum export
%   2. EEG-z-axis wavelet coherence plotting
%   3. SNR calculation based on harmonic-peak removal

clear; clc; close all;

%% Run switches

RUN_EXPORT_FFT        = true;
RUN_WAVELET_COHERENCE = true;
RUN_SNR_CALCULATION   = true;


%% Paths

DATA_FOLDER = fullfile('example_data', 'motion_artifact');

WTC_DATA_FILE = fullfile(DATA_FOLDER, 'ECG_data_example.mat');

COLORMAP_FILE = fullfile(DATA_FOLDER, 'colormap_c.mat');

SNR_DATA_PATH = fullfile(DATA_FOLDER, 'filtered_data');

% Output folders.
OUT_ROOT = fullfile('outputs', 'motion_artifact_analysis_outputs');
FFT_OUT_DIR = fullfile(OUT_ROOT, 'fft_spectra');
WTC_OUT_DIR = fullfile(OUT_ROOT, 'wavelet_coherence');
SNR_OUT_DIR = fullfile(OUT_ROOT, 'snr');

ensure_dir(OUT_ROOT);
ensure_dir(FFT_OUT_DIR);
ensure_dir(WTC_OUT_DIR);
ensure_dir(SNR_OUT_DIR);


%% Common parameters

Fs = 250;                 % Hz
START_SEC = 1;            % start time for FFT/WTC segment
DURATION_SEC = 120;       % segment length for FFT export

START_INDEX = START_SEC * Fs;
SAMPLE_NUM = DURATION_SEC * Fs;
END_INDEX = START_INDEX + SAMPLE_NUM - 1;

% Frequency ranges.
FFT_FREQ_RANGE = [0.2, 10];      % Hz
WTC_FREQ_RANGE = [0.2, 4];       % Hz
SNR_FREQ_RANGE = [1.5, 30];      % Hz

% Wavelet coherence parameters.
VOICES_PER_OCTAVE = 12;
PHASE_DISPLAY_THRESHOLD = 0.75;

% SNR parameters.
PLOT_PSD_FOR_SNR = true;
HARMONIC_ORDERS = [1 2 3 4 5 6];
DEFAULT_HALF_STEP_FREQ = 0.93 / 2;

% Channel mapping.
% First column in each channel pair is gel; second column is paste.
CH_NAMES = ["T8", "P4", "F3", "C3", "F4", "O2"];
CH_TYPES = ["gel", "paste"];

CH = struct( ...
    'T8', [15, 14], ...
    'P4', [13, 12], ...
    'F3', [3, 2], ...
    'C3', [1, 16], ...
    'F4', [5, 4], ...
    'O2', [11, 10], ...
    'z', 7, ...
    'gel', 1, ...
    'paste', 2);


%% FFT harmonic spectrum export

if RUN_EXPORT_FFT
    export_fft_harmonic_spectra( ...
        DATA_FOLDER, FFT_OUT_DIR, Fs, START_INDEX, END_INDEX, ...
        FFT_FREQ_RANGE, CH_NAMES, CH_TYPES, CH);
end


%% EEG-z-axis wavelet coherence

if RUN_WAVELET_COHERENCE
    plot_wavelet_coherence_eeg_z( ...
        WTC_DATA_FILE, COLORMAP_FILE, WTC_OUT_DIR, Fs, START_INDEX, END_INDEX, ...
        WTC_FREQ_RANGE, FFT_FREQ_RANGE, VOICES_PER_OCTAVE, ...
        PHASE_DISPLAY_THRESHOLD, CH_NAMES, CH_TYPES, CH);
end


%% SNR calculation from filtered files

if RUN_SNR_CALCULATION
    calculate_snr_after_harmonic_removal( ...
        SNR_DATA_PATH, SNR_OUT_DIR, Fs, SNR_FREQ_RANGE, ...
        HARMONIC_ORDERS, DEFAULT_HALF_STEP_FREQ, PLOT_PSD_FOR_SNR);
end

fprintf('\nAll selected motion-artifact analysis steps finished.\n');

end


%% Local functions

function ensure_dir(folder_path)
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
end
end


function export_fft_harmonic_spectra(DATA_FOLDER, OUT_DIR, Fs, START_INDEX, END_INDEX, ...
    FFT_FREQ_RANGE, CH_NAMES, CH_TYPES, CH)
% Export FFT amplitude spectra from EEG gel/paste channels and z-axis motion.

fprintf('\n=== FFT harmonic spectrum export ===\n');

file_list = dir(fullfile(DATA_FOLDER, 'ECG_data_*.mat'));

if isempty(file_list)
    warning('No ECG_data_*.mat files were found in:\n%s', DATA_FOLDER);
    return;
end

eeg_out_dir = fullfile(OUT_DIR, 'eeg_fft');
z_out_dir = fullfile(OUT_DIR, 'z_axis_fft');

ensure_dir(eeg_out_dir);
ensure_dir(z_out_dir);

for file_idx = 1:numel(file_list)

    file_name = file_list(file_idx).name;
    file_path = fullfile(DATA_FOLDER, file_name);

    fprintf('Processing FFT file %d/%d: %s\n', file_idx, numel(file_list), file_name);

    S = load(file_path);

    if ~isfield(S, 'data_table')
        warning('data_table was not found in %s. Skipped.', file_name);
        continue;
    end

    data_table = S.data_table;

    if END_INDEX > height(data_table)
        warning('Selected segment exceeds data length in %s. Skipped.', file_name);
        continue;
    end

    data = table2array(data_table(START_INDEX:END_INDEX, 2:end));
    data = detrend(data);

    N = size(data, 1);
    f = (0:N-1) * (Fs / N);
    freq_idx = find(f > FFT_FREQ_RANGE(1) & f <= FFT_FREQ_RANGE(2));

    [~, base_name, ~] = fileparts(file_name);

    % EEG gel/paste FFT
    T_eeg = table(f(freq_idx)', 'VariableNames', {'Frequency'});

    for k = 1:numel(CH_NAMES)
        current_ch_name = CH_NAMES(k);

        for j = 1:numel(CH_TYPES)
            current_ch_type = CH_TYPES(j);
            current_ch = CH.(current_ch_name)(CH.(current_ch_type));

            x = data(:, current_ch);

            % Original normalization.
            y = abs(fft(x) * 1000) / sqrt(Fs * N);

            col_name = sprintf('%s_%s', char(current_ch_name), char(current_ch_type));
            T_eeg.(col_name) = y(freq_idx);
        end
    end

    eeg_csv = fullfile(eeg_out_dir, [base_name, '_EEG_FFT.csv']);
    writetable(T_eeg, eeg_csv);

    % z-axis FFT
    T_z = table(f(freq_idx)', 'VariableNames', {'Frequency'});

    z_data = data(:, CH.z);

    % Original z-axis scaling.
    y_z = abs(fft(z_data / 200 * 1000)) / sqrt(Fs * N);

    T_z.z = y_z(freq_idx);

    z_csv = fullfile(z_out_dir, [base_name, '_z_FFT.csv']);
    writetable(T_z, z_csv);

    % Figures
    fig_eeg = figure('Visible', 'off');
    hold on;
    for c = 2:width(T_eeg)
        plot(T_eeg.Frequency, T_eeg{:, c}, 'LineWidth', 1);
    end
    xlabel('Frequency (Hz)');
    ylabel('FFT amplitude');
    title(['EEG FFT amplitude: ', base_name], 'Interpreter', 'none');
    legend(T_eeg.Properties.VariableNames(2:end), 'Interpreter', 'none', 'Location', 'northeastoutside');
    grid on; box on;
    saveas(fig_eeg, fullfile(eeg_out_dir, [base_name, '_EEG_FFT.png']));
    close(fig_eeg);

    fig_z = figure('Visible', 'off');
    plot(T_z.Frequency, T_z.z, 'k', 'LineWidth', 1.5);
    xlabel('Frequency (Hz)');
    ylabel('FFT amplitude');
    title(['z-axis FFT amplitude: ', base_name], 'Interpreter', 'none');
    grid on; box on;
    saveas(fig_z, fullfile(z_out_dir, [base_name, '_z_FFT.png']));
    close(fig_z);

    fprintf('Saved FFT CSVs and figures for: %s\n', base_name);
end

end


function plot_wavelet_coherence_eeg_z(DATA_FILE, COLORMAP_FILE, OUT_DIR, Fs, START_INDEX, END_INDEX, ...
    WTC_FREQ_RANGE, FFT_FREQ_RANGE, VOICES_PER_OCTAVE, PHASE_DISPLAY_THRESHOLD, ...
    CH_NAMES, CH_TYPES, CH)
% Plot wavelet coherence between each EEG channel and z-axis motion.

fprintf('\n=== EEG-z-axis wavelet coherence ===\n');

if ~isfile(DATA_FILE)
    warning('WTC DATA_FILE was not found:\n%s', DATA_FILE);
    return;
end

S = load(DATA_FILE);

if ~isfield(S, 'data_table')
    warning('data_table was not found in WTC DATA_FILE.');
    return;
end

data_table = S.data_table;

if END_INDEX > height(data_table)
    warning('Selected WTC segment exceeds data length. Skipped WTC.');
    return;
end

if isfile(COLORMAP_FILE)
    C = load(COLORMAP_FILE);
    if isfield(C, 'c')
        cmap = C.c;
    else
        cmap = parula(256);
    end
else
    cmap = parula(256);
end

data = table2array(data_table(:, 2:end));
data = detrend(data);

segment_index = START_INDEX:END_INDEX;
z = data(segment_index, CH.z);

% z-axis FFT reference
N_full = size(data, 1);
f_full = (0:N_full - 1) * (Fs / N_full);
z_full = data(:, CH.z);
y_z = abs(fft(z_full)) / N_full;

fft_idx = find(f_full > FFT_FREQ_RANGE(1) & f_full <= FFT_FREQ_RANGE(2));

[~, peak_rel_idx] = max(y_z(fft_idx));
peak_idx = fft_idx(peak_rel_idx);
peak_freq = f_full(peak_idx);

fig_z = figure('Visible', 'off');
plot(f_full(fft_idx), y_z(fft_idx), 'k', 'LineWidth', 1.5);
hold on;
xline(peak_freq, '--r', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('FFT of z-axis motion channel');
grid on; box on;
saveas(fig_z, fullfile(OUT_DIR, 'FFT_of_z_axis.png'));
close(fig_z);

fprintf('z-axis peak frequency: %.4f Hz\n', peak_freq);

% Wavelet coherence plots
for i = 1:numel(CH_NAMES)
    current_ch_name = CH_NAMES(i);

    for j = 1:numel(CH_TYPES)
        current_ch_type = CH_TYPES(j);
        current_ch = CH.(current_ch_name)(CH.(current_ch_type));

        x = data(segment_index, current_ch);

        ch_name_char = char(current_ch_name);
        ch_type_char = char(current_ch_type);

        fig = figure('Visible', 'off');

        wcoherence( ...
            x, ...
            z, ...
            Fs, ...
            'phasedisplaythreshold', PHASE_DISPLAY_THRESHOLD, ...
            'FrequencyLimits', WTC_FREQ_RANGE, ...
            'VoicesPerOctave', VOICES_PER_OCTAVE);

        colormap(cmap);

        title(sprintf('%s %s Wavelet Coherence', ch_name_char, ch_type_char), ...
            'Interpreter', 'none');

        % Remove white cone-of-influence lines, matching the original workflow.
        coi_lines = findall(gca, 'Type', 'line', 'Color', 'w');
        delete(coi_lines);

        hold on;
        % Keep original reference-line behavior.
        yline(log2(peak_freq), '--', 'LineWidth', 1.5, 'Color', [1, 0, 0]);
        hold off;

        set(fig, 'Position', [100, 100, 390, 190]);

        out_png = fullfile(OUT_DIR, sprintf('%s_%s_Wavelet_Coherence.png', ...
            ch_name_char, ch_type_char));

        print(fig, out_png, '-dpng', '-r300');
        close(fig);

        fprintf('Saved: %s\n', out_png);
    end
end

end


function calculate_snr_after_harmonic_removal(DATA_PATH, OUT_DIR, Fs, SNR_FREQ_RANGE, ...
    HARMONIC_ORDERS, DEFAULT_HALF_STEP_FREQ, PLOT_PSD)
% Calculate SNR by removing walking-frequency harmonic peaks from PSD.

fprintf('\n=== SNR calculation after harmonic-peak removal ===\n');

fs1 = dir(fullfile(DATA_PATH, '*.xlsx'));
fs2 = dir(fullfile(DATA_PATH, '*.xls'));
fs3 = dir(fullfile(DATA_PATH, '*.csv'));

file_struct = [fs1; fs2; fs3];

if isempty(file_struct)
    warning('No .xlsx/.xls/.csv files found in SNR DATA_PATH:\n%s', DATA_PATH);
    return;
end

file_list = fullfile({file_struct.folder}, {file_struct.name});
num_files = numel(file_list);

time_stamp = datestr(now, 'yyyymmdd_HHMMSS');

results = table();
mean_snr_per_file = nan(num_files, 1);

f_low = SNR_FREQ_RANGE(1);
f_high = SNR_FREQ_RANGE(2);

for file_idx = 1:num_files

    fpath = file_list{file_idx};
    fprintf('Analyzing SNR file %d/%d: %s\n', file_idx, num_files, fpath);

    step_freq = DEFAULT_HALF_STEP_FREQ;

    data = readtable(fpath);

    if width(data) < 2
        warning('The file has fewer than two columns and was skipped: %s', fpath);
        continue;
    end

    channel_labels = data.Properties.VariableNames(2:end);
    snr_this_file = nan(numel(channel_labels), 1);

    for ch_idx = 1:numel(channel_labels)

        label = channel_labels{ch_idx};
        parts = split(string(label), "_");

        chan = parts(1);
        type = "NA";
        if numel(parts) >= 2
            type = parts(2);
        end

        eeg = data{:, ch_idx + 1};
        eeg = double(eeg);
        eeg = eeg - mean(eeg, 'omitnan');

        if any(isnan(eeg))
            eeg = fillmissing(eeg, 'linear');
        end

        [psd_raw, f] = pwelch(eeg, hamming(round(Fs * 8)), [], [], Fs);

        freq_mask = (f >= f_low & f <= f_high);
        total_power = trapz(f(freq_mask), psd_raw(freq_mask));

        harmonic_freqs = HARMONIC_ORDERS .* step_freq;
        psd_cleaned = remove_psd_harmonics(f, psd_raw, harmonic_freqs);

        signal_power = trapz(f(freq_mask), psd_cleaned(freq_mask));
        noise_power = total_power - signal_power;

        if signal_power <= 0 || noise_power <= 0
            SNR_dB = NaN;
        else
            SNR_dB = 10 * log10(signal_power / noise_power);
        end

        snr_this_file(ch_idx) = SNR_dB;

        if PLOT_PSD
            fig = figure('Visible', 'off');
            hold on;

            plot(f(freq_mask), psd_raw(freq_mask), 'r', 'LineWidth', 2);
            plot(f(freq_mask), psd_cleaned(freq_mask), 'k', 'LineWidth', 1);

            for h = HARMONIC_ORDERS
                cf = step_freq * h;
                plot([cf, cf], [0, max(psd_raw(freq_mask))], '--k');
            end

            title(sprintf('File %d: %s | %s', file_idx, chan, type), 'Interpreter', 'none');
            xlabel('Frequency (Hz)');
            ylabel('PSD');
            text(f_low, 0, sprintf('S+N: %.4g, S: %.4g, SNR: %.2f dB', ...
                total_power, signal_power, SNR_dB));
            grid on; box on;

            safe_label = regexprep(label, '[^A-Za-z0-9\-_]', '_');
            saveas(fig, fullfile(OUT_DIR, sprintf('PSD_f%03d_%s.png', file_idx, safe_label)));
            close(fig);
        end

        one_row = table( ...
            file_idx, ...
            string(fpath), ...
            string(chan), ...
            string(type), ...
            SNR_dB, ...
            signal_power, ...
            noise_power, ...
            total_power, ...
            'VariableNames', {'FileIndex', 'FilePath', 'Channel', 'Type', ...
                              'SNR_dB', 'SignalPower', 'NoisePower', 'TotalPower'} );

        results = [results; one_row]; %#ok<AGROW>
    end

    mean_snr_per_file(file_idx) = mean(snr_this_file, 'omitnan');
end

out_csv = fullfile(OUT_DIR, ['SNR_results_' time_stamp '.csv']);
writetable(results, out_csv);

Tmean = table( ...
    (1:num_files).', ...
    string(file_list.'), ...
    mean_snr_per_file, ...
    'VariableNames', {'FileIndex', 'FilePath', 'Mean_SNR_dB'} );

out_csv_mean = fullfile(OUT_DIR, ['SNR_mean_per_file_' time_stamp '.csv']);
writetable(Tmean, out_csv_mean);

try
    results_matrix_snr = table( ...
        results.Channel(1:12), ...
        results.Type(1:12), ...
        reshape(results.SNR_dB, [12, num_files]), ...
        'VariableNames', {'Channel', 'Type', 'SNR_dB_all_files'} );

    out_csv_matrix_snr = fullfile(OUT_DIR, ['SNR_matrix_' time_stamp '.csv']);
    writetable(results_matrix_snr, out_csv_matrix_snr);

    results_matrix_signal = table( ...
        results.Channel(1:12), ...
        results.Type(1:12), ...
        reshape(results.SignalPower, [12, num_files]), ...
        'VariableNames', {'Channel', 'Type', 'SignalPower_all_files'} );

    out_csv_matrix_signal = fullfile(OUT_DIR, ['SignalPower_matrix_' time_stamp '.csv']);
    writetable(results_matrix_signal, out_csv_matrix_signal);
catch
    % Skip matrix export if the channel number is not 12 per file.
end

save(fullfile(OUT_DIR, ['SNR_workspace_' time_stamp '.mat']), ...
    'results', 'Tmean', 'DATA_PATH', 'Fs', 'SNR_FREQ_RANGE', ...
    'HARMONIC_ORDERS', 'DEFAULT_HALF_STEP_FREQ');

fprintf('Saved SNR results:\n%s\n%s\n', out_csv, out_csv_mean);

end


function psd_cleaned = remove_psd_harmonics(f, psd_signal, harmonic_freqs)
% Remove sharp harmonic peaks near specified harmonic frequencies and restore
% removed regions by linear interpolation.

N = length(psd_signal);
peak_mask = false(1, N);

window_half_width = 1;

for h = harmonic_freqs

    [~, center] = min(abs(f - h));

    if center == 1 || center == N
        continue;
    end

    left = max(center - window_half_width, 1);
    right = min(center + window_half_width, N);

    [~, local_rel] = max(psd_signal(left:right));
    local_peak_idx = left + local_rel - 1;

    l = local_peak_idx;
    while l > 1 && psd_signal(l - 1) < psd_signal(l)
        l = l - 1;
    end

    r = local_peak_idx;
    while r < N && psd_signal(r + 1) < psd_signal(r)
        r = r + 1;
    end

    if local_peak_idx == l || local_peak_idx == r
        continue;
    end

    psd_local = psd_signal(l:r) - interp1([l, r], psd_signal([l, r]), (l:r)', 'linear');
    peak_val = max(psd_local);
    mean_val = mean(psd_local);

    if peak_val / mean_val > 1.5
        peak_mask((l + 1):(r - 1)) = true;
    end
end

keep_idx = ~peak_mask;
psd_cleaned = interp1(f(keep_idx), psd_signal(keep_idx), f, 'linear');

end
