import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision import transforms

class ECGWaveformDataset(Dataset):
    def __init__(self, data_file, metadata: pd.DataFrame, binary=False, out_size=None, fourier=False,
                 return_stats=False):
        '''
        data_obj: hdf5 file containing the ecg_signals
        metadata: dataframe containing the metadata for the data
        binary: whether or not the problem is a binary classification problem (normal vs abnormal)
        out_size: if not None, will resize the ecg to a specific length
        return_stats: also yield this beat's patient_mean/patient_std, so the
            encoder can z-score each beat by its own record rather than by one
            global constant. off by default -- turning it on changes __getitem__
            from a 2-tuple to a 4-tuple, which every existing training loop
            unpacks as `for data, target in loader`.
        '''

        with h5py.File(data_file, 'r') as f:
            self.data = np.array(f['signal_arrays'])

        self.metadata = metadata
        self.out_size = out_size
        self.binary = binary
        self.fourier = fourier
        self.return_stats = return_stats

        if return_stats and 'patient_mean' not in metadata.columns:
            raise ValueError(
                'return_stats=True needs patient_mean/patient_std columns in the '
                'metadata; regenerate the dataset with create_beat_dataset.py'
            )

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        
        sample = self.metadata.iloc[idx]

        # .iloc on a mixed-dtype frame gives an object/float Series, so once the
        # metadata carries float columns (patient_mean/patient_std) these come
        # back as floats and would fail to index the hdf5 dataset
        waveform = self.data[int(sample.array_index)]
        label = torch.tensor(int(sample.label), dtype=torch.long).repeat(waveform.shape)

        if self.binary:
            label = (label > 0).float()

        if self.out_size is not None:
            t_old = np.linspace(0, 1, len(waveform))
            t_new = np.linspace(0, 1, self.out_size)
            waveform = interp1d(t_old, waveform)(t_new)

        waveform = torch.tensor(waveform)

        if self.fourier:
            waveform = torch.abs(torch.fft.rfft(waveform, dim=-1))

        if self.return_stats:
            return (waveform, label,
                    torch.tensor(sample.patient_mean, dtype=torch.float32),
                    torch.tensor(sample.patient_std, dtype=torch.float32))

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

    
def train_test_val_split_old(
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
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
            transforms.RandomResizedCrop(size=128, scale=(0.9, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
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

        neg_sampled = neg_df.sample(n=num_pos, random_state=42)
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

# the standard inter-patient split for MIT-BIH, from de Chazal et al. (2004).
# the two sets are patient-disjoint and were hand-picked so that the rare
# classes (S, F) appear in similar proportion on both sides -- which
# StratifiedGroupKFold cannot achieve here, because single patients hold more
# of a class than a whole fold's quota (232 alone is ~50% of all S beats,
# and 208 + 213 are ~91% of all F beats).
# assumes the four paced records (102, 104, 107, 217) are already excluded.
DECHAZAL_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
                124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DECHAZAL_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
                212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def dechazal_patient_split(metadata_df):
    # split metadata into train_val (DS1) / test (DS2) using the fixed de Chazal
    # record lists. returns (train_val_df, test_df).
    #
    # this ignores test_prop -- DS1/DS2 is a fixed ~50/50 record split.
    train_val_df = metadata_df[metadata_df['patient'].isin(DECHAZAL_DS1)]
    test_df = metadata_df[metadata_df['patient'].isin(DECHAZAL_DS2)]

    # warn loudly rather than silently dropping data: any patient outside both
    # lists (e.g. a paced record that wasn't excluded) is not assigned anywhere
    unassigned = set(metadata_df['patient'].unique()) - set(DECHAZAL_DS1) - set(DECHAZAL_DS2)
    if unassigned:
        print(f'warning: {len(unassigned)} patient(s) not in DS1 or DS2 and dropped: '
              f'{sorted(unassigned)}')

    return train_val_df, test_df


def stratified_patient_split(metadata_df, test_prop, random_state=0):
    # split metadata into train_val / test where patients are disjoint but the
    # multiclass label proportions are kept roughly equal across the two sides.
    # returns (train_val_df, test_df)
    #
    # StratifiedGroupKFold makes k folds with non-overlapping groups (patients),
    # greedily assigning each patient to whichever fold is currently furthest
    # below its target count for that patient's labels. we take one fold as test.
    n_splits = max(2, int(round(1 / test_prop)))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_val_idx, test_idx = next(sgkf.split(
        metadata_df,
        y=metadata_df['label'],
        groups=metadata_df['patient'],
    ))

    return metadata_df.iloc[train_val_idx], metadata_df.iloc[test_idx]


# keep patients separate for the test set, but train-val can be mixed.
# label proportions are matched across all three splits.
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
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
            transforms.RandomResizedCrop(size=128, scale=(0.9, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]),
        balance=False,
        split='dechazal',
        return_stats=False,
        **dataset_kwargs):
    # same arguments as train_test_val_split, but the test set is both
    # patient-disjoint and label-stratified. val_prop is a fraction of the
    # whole dataset (not of train_val) so the three props should sum to 1.
    #
    # split: how the patient-disjoint test set is chosen.
    #   'stratified' -- StratifiedGroupKFold on the label (previous behaviour).
    #   'dechazal'   -- the fixed DS1/DS2 record lists. test_prop is ignored
    #                   (DS1/DS2 is a fixed ~50/50 record split); train_prop and
    #                   val_prop still set the train:val ratio within DS1.

    metadata_df = pd.read_csv(metadata)

    if balance:
        neg_df = metadata_df[metadata_df["label"] == 0]
        pos_df = metadata_df[metadata_df["label"] != 0]
        num_pos = len(pos_df)

        neg_sampled = neg_df.sample(n=num_pos, random_state=42)
        metadata_df = pd.concat([neg_sampled, pos_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # patient-disjoint test set. fixed random_state so the test set stays
    # identical across runs that vary random_state.
    if split == 'dechazal':
        train_val_df, test_df = dechazal_patient_split(metadata_df)
    elif split == 'stratified':
        train_val_df, test_df = stratified_patient_split(metadata_df, test_prop, random_state=42)
    else:
        raise ValueError(f"split must be 'stratified' or 'dechazal', got {split!r}")

    # train/val can share patients, so stratify exactly at the beat level
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_prop / (train_prop + val_prop),
        stratify=train_val_df['label'],
        random_state=0,
    )

    if dataset_type == 'waveform':
        train_dataset = ECGWaveformDataset(data_file, train_df, binary, out_size,
                                           return_stats=return_stats, **dataset_kwargs)
        val_dataset = ECGWaveformDataset(data_file, val_df, binary, out_size,
                                         return_stats=return_stats, **dataset_kwargs)
        test_dataset = ECGWaveformDataset(data_file, test_df, binary, out_size,
                                          return_stats=return_stats, **dataset_kwargs)

    elif dataset_type == 'spectrogram':
        train_dataset = ECGSpectrogramDataset(train_df, binary, out_size, train_transforms, **dataset_kwargs)
        val_dataset = ECGSpectrogramDataset(val_df, binary, out_size, **dataset_kwargs)
        test_dataset = ECGSpectrogramDataset(test_df, binary, out_size, **dataset_kwargs)

    return train_dataset, val_dataset, test_dataset


def get_beat_bounds(dataset):
    '''
    (start, end) of the labeled beat span inside a window, as window-local indices.

    create_beat_dataset writes these per row: the window is
    [pre_window | window_length | post_window], and only the middle span belongs
    to the labeled beat. slicing a model's output with them supervises exactly
    that beat while the pre/post context stays input-only.

    this replaces the old center_frac heuristic, which assumed the beat sat at
    the window's midpoint and guessed its extent as a fraction.
    '''

    metadata = dataset.metadata

    if 'window_start' not in metadata.columns:
        raise ValueError(
            'dataset metadata has no window_start/window_end columns; regenerate '
            'it with create_beat_dataset.py'
        )

    starts = metadata['window_start'].unique()
    ends = metadata['window_end'].unique()

    # a single slice is applied to the whole batch, so a dataset mixing
    # geometries would silently supervise the wrong span for some of its beats
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            f'dataset mixes window geometries (starts={starts}, ends={ends}); '
            'all beats must share one layout'
        )

    return int(starts[0]), int(ends[0])


def get_dataset_statistics(dataset):
    # returns the mean and standard deviation of the dataset (useful for normalization)
    all_samples = [dataset.__getitem__(i)[0] for i in range(len(dataset))]
    return np.mean(all_samples), np.std(all_samples)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_file = '../beats/data.hdf5'
    metadata = '../beats/data.csv'
    out_size = None

    metadata_df = pd.read_csv(metadata)

    
    print(metadata_df['label'].value_counts())

    # for rs in [1]:
    #     train_dataset, val_dataset, test_dataset = train_test_val_split_old(data_file, metadata, 0.6, 0.2, 0.2, balance=True, random_state=rs, fourier=True)
    #     print(train_dataset.metadata['label'].value_counts() / len(train_dataset), val_dataset.metadata['label'].value_counts() / len(val_dataset), test_dataset.metadata['label'].value_counts() / len(test_dataset))

    for rs in [0, 1, 2]:
        train_dataset, val_dataset, test_dataset = train_test_val_split(data_file, metadata, 0.6, 0.2, 0.2, balance=True, random_state=rs, fourier=True)
        print(train_dataset.metadata['label'].value_counts() / len(train_dataset), val_dataset.metadata['label'].value_counts() / len(val_dataset), test_dataset.metadata['label'].value_counts() / len(test_dataset))
    # print(np.where(train_dataset.metadata["label"] == 1)[0])

    # print(train_dataset.metadata['label'].value_counts())
    # print(val_dataset.metadata['label'].value_counts())
    # print(test_dataset.metadata['label'].value_counts())

    # waveform, label = train_dataset.__getitem__(399)

    # plt.plot(waveform)
    # plt.title(label)
    # plt.show()
