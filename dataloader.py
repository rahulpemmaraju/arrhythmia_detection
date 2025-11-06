import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision import transforms

class ECGWaveformDataset(Dataset):
    def __init__(self, data_file, metadata: pd.DataFrame, binary=False, out_size=None, fourier=False):
        '''
        data_obj: hdf5 file containing the ecg_signals
        metadata: dataframe containing the metadata for the data 
        binary: whether or not the problem is a binary classification problem (normal vs abnormal)
        out_size: if not None, will resize the ecg to a specific length
        '''

        with h5py.File(data_file, 'r') as f:
            self.data = np.array(f['signal_arrays'])

        self.metadata = metadata
        self.out_size = out_size
        self.binary = binary
        self.fourier = fourier

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        
        sample = self.metadata.iloc[idx]

        waveform = self.data[sample.array_index]
        label = torch.tensor(sample.label, dtype=torch.long)

        if self.binary:
            label = torch.tensor(label > 0, dtype=torch.float32)

        if self.out_size is not None:
            t_old = np.linspace(0, 1, len(waveform))
            t_new = np.linspace(0, 1, self.out_size)
            waveform = interp1d(t_old, waveform)(t_new)

        waveform = torch.tensor(waveform)

        if self.fourier:
            waveform = torch.abs(torch.fft.rfft(waveform, dim=-1))

        return waveform, label
    
class ECGSpectrogramDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, binary=False, out_size=(128, 128), random_transforms=None):
        '''
        data_obj: hdf5 file containing the ecg_signals
        metadata: dataframe containing the metadata for the data 
        binary: whether or not the problem is a binary classification problem (normal vs abnormal)
        out_size: if not None, will resize the ecg to a specific length
        '''

        self.metadata = metadata
        self.out_size = out_size
        self.binary = binary

        self.resize = transforms.Resize(out_size)
        self.random_transforms = random_transforms

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        
        sample = self.metadata.iloc[idx]

        img = decode_image(sample['spectrogram_path']).to(torch.float32) / 255.0
        img = self.resize(img)

        if self.random_transforms is not None:
            img = self.random_transforms(img)

        label = sample.label

        if self.binary:
            label = torch.tensor(label > 0, dtype=torch.float32)

        return img, label

    
def train_test_val_split(
        data_file, 
        metadata, 
        train_prop, 
        val_prop, 
        test_prop, 
        binary=False, 
        out_size=None, 
        random_state=0, 
        dataset_type='waveform', 
        train_transforms=transforms.Compose([
            transforms.RandomRotation(degrees=15),          # small random rotation
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),  # translate + scale
            transforms.RandomResizedCrop(size=128, scale=(0.9, 1.0)),  # random crop + resize
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # blur
        ]),
        balance=False,
        **dataset_kwargs):
    # split the data into train/val/test splits at a patient level -> return ECGWaveformDataset objects for each one
    # data_file: path to file with raw data
    # metadata: path to metadata file
    # train_prop: proportion of data to use for training
    # val_prop: proportion of data to use for validation
    # test_prop: proportion of data to use for testing
    # binary: whether or not the problem is a binary classification problem (normal vs abnormal)
    # out_size: used to resize ecg to specific length (default None -> no resizing which is what we probably want)
    # dataset_type: either "waveform" (for raw waveforms) or "spectrogram" for spectrogram images
    # train_transforms: only for spectrogram -> used for training augmentatino
    # random_state: ensures consistent splitting of data
    # balance: tries to make the dataset slightly more balanced

    metadata_df = pd.read_csv(metadata)

    # balance dataset by ensuring there are equal number of positive and negative samples
    if balance:
        neg_df = metadata_df[metadata_df["label"] == 0]
        pos_df = metadata_df[metadata_df["label"] != 0]
        num_pos = len(pos_df)

        neg_sampled = neg_df.sample(n=num_pos, random_state=random_state)
        metadata_df = pd.concat([neg_sampled, pos_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    patients = metadata_df['patient'].unique()
    train_val_patients, test_patients = train_test_split(patients, test_size=test_prop, random_state=42)
    
    train_patients, val_patients = train_test_split(train_val_patients, test_size=val_prop, random_state=random_state)

    train_df = metadata_df[metadata_df['patient'].isin(train_patients)]
    val_df = metadata_df[metadata_df['patient'].isin(val_patients)]
    test_df = metadata_df[metadata_df['patient'].isin(test_patients)]

    if dataset_type == 'waveform':
        train_dataset = ECGWaveformDataset(data_file, train_df, binary, out_size, **dataset_kwargs)
        val_dataset = ECGWaveformDataset(data_file, val_df, binary, out_size, **dataset_kwargs)
        test_dataset = ECGWaveformDataset(data_file, test_df, binary, out_size, **dataset_kwargs)

    elif dataset_type == 'spectrogram':
        train_dataset = ECGSpectrogramDataset(train_df, binary, out_size, train_transforms, **dataset_kwargs)
        val_dataset = ECGSpectrogramDataset(val_df, binary, out_size, **dataset_kwargs)
        test_dataset = ECGSpectrogramDataset(test_df, binary, out_size, **dataset_kwargs)

    return train_dataset, val_dataset, test_dataset

def get_dataset_statistics(dataset):
    # returns the mean and standard deviation of the dataset (useful for normalization)
    all_samples = [dataset.__getitem__(i)[0] for i in range(len(dataset))]
    return np.mean(all_samples), np.std(all_samples)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_file = '../beat_neurokit_1.hdf5'
    metadata = '../beat_neurokit_1.csv'
    out_size = None

    
    train_dataset, val_dataset, test_dataset = train_test_val_split(data_file, metadata, 0.6, 0.2, 0.2, balance=True, random_state=42, fourier=True)
    print(get_dataset_statistics(train_dataset))
    # print(np.where(train_dataset.metadata["label"] == 1)[0])

    # print(train_dataset.metadata['label'].value_counts())
    # print(val_dataset.metadata['label'].value_counts())
    # print(test_dataset.metadata['label'].value_counts())

    # waveform, label = train_dataset.__getitem__(399)

    # plt.plot(waveform)
    # plt.title(label)
    # plt.show()