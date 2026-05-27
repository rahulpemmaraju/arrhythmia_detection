import argparse
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
BEAT_TO_ENCODING = {
    '!': 4,
    '"': -1,
    '+': -1,
    '/': 4,
    'A': 1,
    'E': 2,
    'F': 3,
    'J': 1,
    'L': 0,
    'N': 0,
    'Q': 4,
    'R': 0,
    'S': 1,
    'V': 2,
    '[': 4,
    ']': 4,
    'a': 1,
    'e': 0,
    'f': 0,
    'j': 0,
    'x': 4,
    '|': -1,
    '~': -1,
}

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

def get_windowed_data(signal_array, annotation, window_length):
    '''
    window the data around a beat
    signal_array: cleaned ecg signal
    window_length: length of window (in seconds)
    stride: stride for windowing (in seconds)
    fs: sampling rate of the signal
    returns: list of windows, list of labels
    '''

    windowed_arrays = []
    windowed_labels = []

    beat_encoding = np.array([BEAT_TO_ENCODING[x] for x in annotation.symbol])
    beat_samples = np.array(annotation.sample)
    mask = np.where(beat_encoding > -1)

    filtered_encoding = beat_encoding[mask]
    filtered_beats = beat_samples[mask]


    window_len = window_length * annotation.fs

    for i, x in enumerate(filtered_beats):
        start = int(x - window_len // 2)
        end = int(x + window_len // 2)

        if start >= 0 and end < len(signal_array):
            windowed_labels.append(int(filtered_encoding[i]))
            windowed_arrays.append(signal_array[start:end])

    return windowed_arrays, windowed_labels
        

def preprocess_ecg(signal_array, method, fs):
    '''
    returns a preprocessed ecg
    signal_array: 1D array of the ecg signal
    method: preprocessing algorithm to use (from neurokit for now)
    fs: sampling rate of the signal
    '''

    return nk.ecg_clean(signal_array, sampling_rate=fs, method=method)


def process_patient(patient_path, window_length=1, preprocessing_method='neurokit'):
    '''
    does all the processing for a patient and returns the windowed data and metadata dict
    patient_path: path to ecg
    dataset: either 'nsr' or 'arr' indicating which dataset the patient belongs to
    window_length: segment length for segmenting the data
    stride: how far to jump (in seconds) to get the next window of data
    preprocessing_method: the neurokit preprocessing algorithm to use for the ecg

    returns: list of arrays corresponding to windowed ecg and list of dicts corresponding to dataframe elements
    '''

    record = wfdb.rdrecord(patient_path)
    annotation = wfdb.rdann(patient_path, 'atr')

    fs = record.fs
    signal_array = record.p_signal[:, 0] # only using lead 1 here
    

    signal_array = preprocess_ecg(signal_array, preprocessing_method, fs)
    patient_arrays, patient_labels = get_windowed_data(signal_array, annotation, window_length)

    patient_metadata = [{'patient': annotation.record_name, 'fs': fs, 'label': l}  for l in patient_labels]

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
    parser.add_argument('--raw_data_folder', type=str, help='Path to processed data files.')
    parser.add_argument('--processed_data_folder', type=str, help='Path to processed data files.')
    parser.add_argument('--window_length', type=float, default=1,
                        help='Window length in seconds.')
    parser.add_argument('--preprocessing_method', type=str, default='neurokit',
                        help='NeuroKit preprocessing method to use.')
    return parser.parse_args()


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

    arrhythmia_paths = ['{}/{}'.format(args.raw_data_folder, x) for x in arrythmia_samples]

    for path in tqdm(arrhythmia_paths):
        patient_arrays, patient_metadata = process_patient(
            path,
            args.window_length,
            args.preprocessing_method,
        )
        metadata_dicts += patient_metadata
        append_vlen(patient_arrays, hdf5_path)

    metadata_df = pd.DataFrame(metadata_dicts)
    metadata_df['array_index'] = metadata_df.index

    metadata_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
