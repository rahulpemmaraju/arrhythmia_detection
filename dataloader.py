import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

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


def train_test_val_split(
        data_file,
        metadata,
        train_prop,
        val_prop,
        test_prop,
        binary=False,
        out_size=None,
        random_state=0,
        balance=False,
        return_stats=False,
        **dataset_kwargs):
    # patient-disjoint test set (de Chazal DS2), with train/val split from DS1.
    #
    # test_prop is ignored: DS1/DS2 is a fixed ~50/50 record split. train_prop and
    # val_prop set the train:val ratio within DS1 (0.6:0.2 = 75:25).
    #
    # the split does not depend on random_state: balancing, the DS1/DS2 split and
    # the train/val split all use fixed seeds, so every model seed sees the same
    # data and seeds change only weight initialization and batch order.

    metadata_df = pd.read_csv(metadata)

    if balance:
        neg_df = metadata_df[metadata_df["label"] == 0]
        pos_df = metadata_df[metadata_df["label"] != 0]
        num_pos = len(pos_df)

        neg_sampled = neg_df.sample(n=num_pos, random_state=42)
        metadata_df = pd.concat([neg_sampled, pos_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    train_val_df, test_df = dechazal_patient_split(metadata_df)

    # train/val can share patients, so stratify exactly at the beat level
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_prop / (train_prop + val_prop),
        stratify=train_val_df['label'],
        random_state=0,
    )

    train_dataset = ECGWaveformDataset(data_file, train_df, binary, out_size,
                                       return_stats=return_stats, **dataset_kwargs)
    val_dataset = ECGWaveformDataset(data_file, val_df, binary, out_size,
                                     return_stats=return_stats, **dataset_kwargs)
    test_dataset = ECGWaveformDataset(data_file, test_df, binary, out_size,
                                      return_stats=return_stats, **dataset_kwargs)

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
    # print the split sizes and class balance of a processed dataset
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../beats'

    splits = train_test_val_split(f'{data_dir}/data.hdf5', f'{data_dir}/data.csv',
                                  0.6, 0.2, 0.2, binary=True, balance=True)

    for name, dataset in zip(('train', 'val', 'test'), splits):
        labels = dataset.metadata['label']
        print(f'{name}: {len(dataset)} beats, {(labels > 0).mean():.3f} arrhythmic')
