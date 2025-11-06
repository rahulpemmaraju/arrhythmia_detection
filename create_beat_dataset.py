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

from config import ARR_DATA_DIR, NSR_DATA_DIR, BEAT_TO_ENCODING, NSR_TO_ENCODING

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


def process_patient(patient_path, img_folder, dataset='nsr', window_length=1, preprocessing_method='neurokit'):
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

if __name__ == '__main__':

    window_length = 1
    preprocessing_method = 'neurokit'

    dataset_name = f'beat_{preprocessing_method}_{window_length}'

    h5_path = os.path.join('/Users/rahul/Documents/G1/BrainInspiredComputing/TermProject', f'{dataset_name}.hdf5')
    csv_path = os.path.join('/Users/rahul/Documents/G1/BrainInspiredComputing/TermProject', f'{dataset_name}.csv')
    img_folder = os.path.join('/Users/rahul/Documents/G1/BrainInspiredComputing/TermProject', f'{dataset_name}')
    os.makedirs(img_folder, exist_ok=True)

    metadata_dicts = []

    with h5py.File(h5_path, 'w') as f:
        dt = h5py.vlen_dtype(np.dtype('float32'))
        f.create_dataset('signal_arrays', shape=(0,), maxshape=(None,), dtype=dt)

    
    print('--- MIT BIH ARR DATASET --- ')
    # arrhythmia data
    f = open(os.path.join(ARR_DATA_DIR, 'RECORDS'))
    arrythmia_samples = [x.strip() for x in f.readlines()]

    arrhythmia_paths = ['{}/{}'.format(ARR_DATA_DIR, x) for x in arrythmia_samples]

    for path in tqdm(arrhythmia_paths):
        patient_arrays, patient_metadata = process_patient(path, img_folder, 'arr', window_length, preprocessing_method)
        metadata_dicts += patient_metadata
        append_vlen(patient_arrays, h5_path)

    # print('--- MIT BIH NSR DATAST --- ')
    # # normal data
    # f = open(os.path.join(NSR_DATA_DIR, 'RECORDS'))
    # arrythmia_samples = [x.strip() for x in f.readlines()]

    # arrhythmia_paths = ['{}/{}'.format(NSR_DATA_DIR, x) for x in arrythmia_samples]

    # for path in tqdm(arrhythmia_paths):
    #     patient_arrays, patient_metadata = process_patient(path, 'nsr', window_length, stride, preprocessing_method)
    #     metadata_dicts += patient_metadata
    #     append_vlen(patient_arrays, h5_name)
    #     break

    metadata_df = pd.DataFrame(metadata_dicts)
    metadata_df['array_index'] = metadata_df.index

    metadata_df.to_csv(csv_path, index=False)
