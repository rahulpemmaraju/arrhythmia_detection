import argparse
import json
import os

import h5py
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import neurokit2 as nk
from scipy import stats
import scipy.signal as signal

from config import ARR_DATA_DIR

# USE THE AAMI EC57 STANDARD: https://arxiv.org/html/2503.07276v1
# IGNORE CLASS Q (PACED BEATS, V FLUTTER)
BEAT_TO_ENCODING = {
    '!': -1,
    '"': -1,
    '+': -1,
    '/': -1,
    'A': 1,
    'E': 2,
    'F': 3,
    'J': 1,
    'L': 0,
    'N': 0,
    'Q': -1,
    'R': 0,
    'S': 1,
    'V': 2,
    '[': -1,
    ']': -1,
    'a': 1,
    'e': 0,
    'f': -1,
    'j': 0,
    'x': -1,
    '|': -1,
    '~': -1,
}

AAMI_EXCLUDED = {'102', '104', '107', '217'}  # paced records, per AAMI EC57

ARRHYTHMIA_TO_ENCODING = {
    '(AB': 2,
    '(AFIB': 1,
    '(AFIB': 1,
    '(AFL': 2,
    '(B': 3,
    '(BII': 5,
    '(IVR': 3,
    '(N': 0,
    '(NOD': 2,
    '(P': 4,
    '(PREX': 4,
    '(SBR': 0,
    '(SVTA': 2,
    '(T': 3,
    '(VFL': 3,
    '(VT': 3,
    'MISSB': -1,
    'PSE': -1,
    'TS': -1, 
    '': -1,
}

NSR_TO_ENCODING = {
    'F': -1, 
    'J': -1, 
    'N': 0, 
    'S': -1, 
    'V': -1, 
    '|': -1, 
    '~': -1
}


'''
this script will be used to preprocess -> slice -> label -> save ecg data
for labeling... a window will be labeled based on beat-level annotation
data will be preprocessed using the neurokit method: highpass filter + bandpass filter

output format: 
    pandas dataframe: sample metadata (patient/record, label, sampling rate)
    hdf5: sample array

'''

def window_geometry(window_length, pre_window, post_window, fs):
    '''
    sample offsets describing a window's layout, all relative to the window itself.

    the extracted window is  [pre_window | window_length | post_window]  in
    seconds, so it is three concatenated spans:

        0 .............. window_start ....... window_end ......... total
        |<- pre context ->|<--- the beat --->|<- post context ->|

    window_length is the span *belonging to the labeled beat*, centered on the R
    annotation. pre_window and post_window are extra context: fed to the model as
    input, but outside the labeled span. that is what replaces center_frac --
    instead of guessing a central fraction, the beat's own bounds are recorded.

    returns (window_start, window_end, total, half) as sample counts, where half
    is the beat span's half-width used to place the R peak.
    '''

    beat_len = int(round(window_length * fs))
    pre_len = int(round(pre_window * fs))
    post_len = int(round(post_window * fs))

    # the beat span is 2*half so the R annotation lands exactly on its midpoint
    half = beat_len // 2

    window_start = pre_len
    window_end = pre_len + 2 * half
    total = window_end + post_len

    return window_start, window_end, total, half


def get_windowed_data(signal_array, annotation, window_length, pre_window, post_window, fs,
                      downsample_factor=1):
    '''
    window the data around a beat.

    signal_array: cleaned ecg signal
    window_length: seconds belonging to the labeled beat, centered on the R annotation
    pre_window: extra seconds of context before that span
    post_window: extra seconds of context after that span
    fs: sampling rate of the signal (after downsampling)
    downsample_factor: factor the signal was downsampled by, used to rescale beat indices

    pre_window = post_window = 0 reduces to the original behaviour: a window of
    window_length centered on the beat, with the beat span covering all of it.

    returns: list of windows, list of labels, list of beat sample positions in the
        record. the window-local bounds are identical for every beat, so they are
        returned once by window_geometry rather than per beat.
    '''

    windowed_arrays = []
    windowed_labels = []
    windowed_beats = []

    beat_encoding = np.array([BEAT_TO_ENCODING[x] for x in annotation.symbol])
    beat_samples = np.array(annotation.sample) // downsample_factor
    mask = np.where(beat_encoding > -1)

    filtered_encoding = beat_encoding[mask]
    filtered_beats = beat_samples[mask]

    window_start, window_end, total, half = window_geometry(
        window_length, pre_window, post_window, fs)

    for i, x in enumerate(filtered_beats):
        # walk back from the R peak through the beat's own half-span and the pre context
        start = int(x - half - window_start)
        end = start + total

        # every window must be the same length for the hdf5 + batching to work,
        # so beats too close to either edge of the record are dropped as before
        if start >= 0 and end <= len(signal_array):
            windowed_labels.append(int(filtered_encoding[i]))
            windowed_arrays.append(signal_array[start:end])
            windowed_beats.append(int(x))

    return windowed_arrays, windowed_labels, windowed_beats
        

def preprocess_ecg(signal_array, method, fs):
    '''
    returns a preprocessed ecg
    signal_array: 1D array of the ecg signal
    method: preprocessing algorithm to use (from neurokit for now)
    fs: sampling rate of the signal
    '''

    return nk.ecg_clean(signal_array, sampling_rate=fs, method=method)


def downsample_ecg(signal_array, downsample_factor):
    '''
    downsamples an ecg by an integer factor with an anti-aliasing filter
    signal_array: 1D array of the ecg signal
    downsample_factor: integer factor to downsample by
    '''

    if downsample_factor == 1:
        return signal_array

    return signal.resample_poly(signal_array, up=1, down=downsample_factor)


def process_patient(patient_path, window_length=1, pre_window=0.0, post_window=0.0,
                    preprocessing_method='neurokit', downsample_factor=1):
    '''
    does all the processing for a patient and returns the windowed data and metadata dict
    patient_path: path to ecg
    dataset: either 'nsr' or 'arr' indicating which dataset the patient belongs to
    window_length: seconds belonging to the labeled beat, centered on the R annotation
    pre_window: extra seconds of context before that span
    post_window: extra seconds of context after that span
    preprocessing_method: the neurokit preprocessing algorithm to use for the ecg
    downsample_factor: integer factor to downsample the ecg by before preprocessing

    returns: list of arrays corresponding to windowed ecg and list of dicts corresponding to dataframe elements
    '''

    record = wfdb.rdrecord(patient_path)
    annotation = wfdb.rdann(patient_path, 'atr')

    signal_array = record.p_signal[:, 0] # only using lead 1 here

    signal_array = downsample_ecg(signal_array, downsample_factor)
    fs = record.fs // downsample_factor

    signal_array = preprocess_ecg(signal_array, preprocessing_method, fs)
    patient_arrays, patient_labels, patient_beats = get_windowed_data(
        signal_array, annotation, window_length, pre_window, post_window, fs, downsample_factor
    )

    window_start, window_end, _, _ = window_geometry(window_length, pre_window, post_window, fs)

    # per-record statistics, for patient-specific normalization at training time.
    #
    # a single global mean/std over the whole training set leaves each patient's
    # amplitude scaling and baseline offset in the input -- lead placement, gain,
    # body habitus. a model can fit that on DS1, but it is simply wrong on DS2,
    # so it is a plausible part of the inter-patient generalization gap.
    #
    # taken over the full cleaned record rather than pooled from the windows:
    # the windows overlap (a 2s window per beat, beats <1s apart), so pooling
    # them would double-count the overlap regions and silently weight the
    # statistic toward densely-beat regions.
    record_mean = float(np.mean(signal_array))
    record_std = float(np.std(signal_array))

    # window_start/window_end are WINDOW-LOCAL: they index into the extracted
    # window, marking the span that belongs to the labeled beat. slicing a model's
    # output with them supervises exactly that beat and leaves the pre/post context
    # as input only -- which is what center_frac was approximating.
    # beat_sample is the R peak's position in the record, kept for RR calculations.
    patient_metadata = [
        {
            'patient': annotation.record_name,
            'fs': fs,
            'label': l,
            'patient_mean': record_mean,
            'patient_std': record_std,
            'beat_sample': beat,
            'window_start': window_start,
            'window_end': window_end,
        }
        for l, beat in zip(patient_labels, patient_beats)
    ]

    return patient_arrays, patient_metadata


def append_vlen(signal_arrays, h5file):
    # appends an individal signal array to an existing h5 file

    with h5py.File(h5file, 'a') as f:
        d_arrays = f['signal_arrays']

        n_current = d_arrays.shape[0]
        n_new = len(signal_arrays)

        d_arrays.resize((n_current + n_new,))

        for i, arr in enumerate(signal_arrays):
            d_arrays[n_current + i] = arr

def parse_args():
    parser = argparse.ArgumentParser(description='Create beat datasets from ECG recordings.')
    parser.add_argument('--raw_data_folder', type=str, help='Path to raw data.')
    parser.add_argument('--processed_data_folder', type=str, help='Path to processed data files.')
    parser.add_argument('--window_length', type=float, default=1,
                        help='Seconds belonging to the labeled beat, centered on the R '
                             'annotation. This is the span the loss is computed over.')
    parser.add_argument('--pre_window', type=float, default=0.0,
                        help='Extra seconds of context before the beat span. Input only, '
                             'outside the labeled region.')
    parser.add_argument('--post_window', type=float, default=0.0,
                        help='Extra seconds of context after the beat span. Input only, '
                             'outside the labeled region. 0 means no lookahead beyond '
                             'the beat span itself.')
    parser.add_argument('--preprocessing_method', type=str, default='neurokit',
                        help='NeuroKit preprocessing method to use.')
    parser.add_argument('--downsample_factor', type=int, default=1,
                        help='Integer factor to downsample the ECG by (1 keeps the original rate).')
    args = parser.parse_args()

    if args.downsample_factor < 1:
        parser.error('--downsample_factor must be a positive integer.')

    if args.window_length <= 0:
        parser.error('--window_length must be positive.')

    if args.pre_window < 0 or args.post_window < 0:
        parser.error('--pre_window and --post_window must be non-negative.')

    return args


def main():
    args = parse_args()

    hdf5_path = os.path.join(args.processed_data_folder, f'data.hdf5')
    csv_path = os.path.join(args.processed_data_folder, f'data.csv')

    os.makedirs(args.processed_data_folder, exist_ok=True)

    metadata_dicts = []

    with h5py.File(hdf5_path, 'w') as f:
        dt = h5py.vlen_dtype(np.dtype('float32'))
        f.create_dataset('signal_arrays', shape=(0,), maxshape=(None,), dtype=dt)

    print('--- MIT BIH ARR DATASET --- ')
    with open(os.path.join(args.raw_data_folder, 'RECORDS')) as f:
        arrythmia_samples = [x.strip() for x in f.readlines()]
    arrythmia_samples = [x for x in arrythmia_samples if x not in AAMI_EXCLUDED] # remove paced samples

    arrhythmia_paths = ['{}/{}'.format(args.raw_data_folder, x) for x in arrythmia_samples]

    for path in tqdm(arrhythmia_paths):
        patient_arrays, patient_metadata = process_patient(
            path,
            args.window_length,
            args.pre_window,
            args.post_window,
            args.preprocessing_method,
            args.downsample_factor,
        )
        metadata_dicts += patient_metadata
        append_vlen(patient_arrays, hdf5_path)

    metadata_df = pd.DataFrame(metadata_dicts)
    metadata_df['array_index'] = metadata_df.index

    metadata_df.to_csv(csv_path, index=False)

    # sidecar describing the window geometry, so downstream code can read where the
    # labeled beat sits inside a window instead of assuming it is centered.
    # beat_offset is constant across every window in a dataset (it is pre_window
    # in samples), but it is written per-beat in the csv too so a mixed dataset
    # would still be self-describing.
    fs_values = sorted(metadata_df['fs'].unique().tolist())
    fs = fs_values[0] if len(fs_values) == 1 else fs_values

    window_start, window_end, total, half = window_geometry(
        args.window_length, args.pre_window, args.post_window, fs)

    window_config = {
        'window_length_s': args.window_length,
        'pre_window_s': args.pre_window,
        'post_window_s': args.post_window,
        'downsample_factor': args.downsample_factor,
        'preprocessing_method': args.preprocessing_method,
        'fs': fs,
        'total_samples': total,       # length of each extracted window
        'window_start': window_start, # window-local start of the labeled beat span
        'window_end': window_end,     # window-local end of the labeled beat span
        'beat_offset': window_start + half,  # window-local index of the R peak
        'n_beats': len(metadata_df),
    }

    with open(os.path.join(args.processed_data_folder, 'window_config.json'), 'w') as f:
        json.dump(window_config, f, indent=2)

    print('window config:', window_config)


if __name__ == '__main__':
    main()
