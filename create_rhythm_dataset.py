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

from config import ARR_DATA_DIR, NSR_DATA_DIR, ARRHYTHMIA_TO_ENCODING, NSR_TO_ENCODING

'''
this script will be used to preprocess -> slice -> label -> save ecg data
for labeling... a window will only be considered arrhythmic if a majority is arrhythmic (the most common label in windows with multiple labels will be considered)
data will be preprocessed using the neurokit method: highpass filter + bandpass filter

output format: 
    pandas dataframe: sample metadata (patient/record, label, sampling rate)
    hdf5: sample array

'''

def get_per_sample_label(signal_array, annotation, dataset):
    # signal_array: 1d array of ECG data
    # annotation: annotation file for the corresponding signal_arrya
    # dataset: either "nsr" or "arr": dictates how the data will be labeled

    if dataset == 'nsr':
        symbol_note = annotation.symbol

        per_sample_beat_annotation = - np.ones_like(signal_array)

        start_indeces = annotation.sample
        end_indeces = np.concatenate([annotation.sample[1:], [len(per_sample_beat_annotation)]])

        for si, ei, ba in zip(start_indeces, end_indeces, symbol_note):
            per_sample_beat_annotation[si:ei] = NSR_TO_ENCODING[ba]

        return per_sample_beat_annotation

    elif dataset == 'arr':
        aux_note = np.char.replace(np.array(annotation.aux_note, dtype=str), '\x00', '')
        per_beat_rhythm_annotation = np.empty_like(aux_note)

        start_indeces = np.where(aux_note != '')[0]
        end_indeces = np.concatenate([np.where(aux_note != '')[0][1:], [len(aux_note)]])
        rhythm_vals = aux_note[np.where(aux_note != '')]

        for si, ei, ba in zip(start_indeces, end_indeces, rhythm_vals):
            per_beat_rhythm_annotation[si:ei] = ba

        per_sample_rhythm_annotation = - np.ones_like(signal_array)

        start_indeces = annotation.sample
        end_indeces = np.concatenate([annotation.sample[1:], [len(per_sample_rhythm_annotation)]])

        for si, ei, ba in zip(start_indeces, end_indeces, per_beat_rhythm_annotation):
            per_sample_rhythm_annotation[si:ei] = ARRHYTHMIA_TO_ENCODING[ba]

        return per_sample_rhythm_annotation
    
def get_windowed_data(signal_array, label_array, window_length, stride, fs):
    '''
    window the data using a sliding window approach
    signal_array: cleaned ecg signal
    window_length: length of window (in seconds)
    stride: stride for windowing (in seconds)
    fs: sampling rate of the signal
    returns: list of windows, list of labels
    '''

    windowed_arrays = []
    windowed_labels = []

    for i in range(0, len(signal_array), fs*window_length):

        start = i
        end = i + window_length * fs

        if end > len(signal_array):
            break

        window_ecg = signal_array[start:end]
        window_all_labels = label_array[start:end]

        window_label = int(stats.mode(window_all_labels).mode)

        if not window_label == -1:
            windowed_arrays.append(window_ecg)
            windowed_labels.append(window_label)

    return windowed_arrays, windowed_labels
        

def preprocess_ecg(signal_array, method, fs):
    '''
    returns a preprocessed ecg
    signal_array: 1D array of the ecg signal
    method: preprocessing algorithm to use (from neurokit for now)
    fs: sampling rate of the signal
    '''

    return nk.ecg_clean(signal_array, sampling_rate=fs, method=method)

def get_spectrograms(signal_arrays, fs, patient_name, out_dir):
    '''
    computes spectrograms for list of signal_arrays, saves as png files, and returns file names
    signal_array: 1D array of the ecg signal
    fs: sampling rate of the signal
    patient_name: name of the patient (for saving output file)
    '''

    out_paths = []

    for i, signal_array in enumerate(signal_arrays):
        out_path = os.path.join(out_dir, f'{patient_name}_{i}.png')
        f, t_spec, Sxx  = signal.spectrogram(signal_array, nperseg=256, noverlap=245)
        Sxx_log = 10 * np.log10(Sxx + 1e-8)
        Sxx_norm = (255 * (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())).astype(np.uint8)
        img = Image.fromarray(Sxx_norm)
        img.save(out_path)

        out_paths.append(out_path)
        
    return out_paths

def process_patient(patient_path, img_folder, dataset='nsr', window_length=5, stride=2, preprocessing_method='neurokit'):
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
    label_array = get_per_sample_label(signal_array, annotation, dataset)

    patient_arrays, patient_labels = get_windowed_data(signal_array, label_array, window_length, stride, fs)
    spectrogram_paths = get_spectrograms(patient_arrays, fs, annotation.record_name, img_folder)

    patient_metadata = [

        {'patient': annotation.record_name, 
         'fs': fs, 'label': l, 
         'spectrogram_path': spectrogram_paths[i]} 

    for i, l in enumerate(patient_labels)]

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

    window_length = 5
    stride = 2
    preprocessing_method = 'neurokit'

    dataset_name = f'arr_only_{preprocessing_method}_{window_length}_{stride}'

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
        patient_arrays, patient_metadata = process_patient(path, img_folder, 'arr', window_length, stride, preprocessing_method)
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
