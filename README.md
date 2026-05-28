## Hierarchically Gated Spiking Neural Networks for Interpretable Arrhythmia Detection

This repository contains the code for the following publication:
(link to come)

## Installation
Clone this repository and install the requirements by running the following commands in your terminal:

```
git clone https://github.com/rahulpemmaraju/arrhythmia_detection.git
```
Create a virtual environment and install all project dependencies.

```
conda create --name hgrsnn python=3.12.0
conda activate hgrsnn
pip install -r requirements.txt
```

## Data Preparation
Download data from [MIT-BIH Database](https://www.physionet.org/content/mitdb/1.0.0/)

Preprocess data for model training using the following command:
```
python create_beat_dataset.py --raw_data_folder={path to raw data} --processed_data_folder={path to processed data}
```

## Train/Evaluate Arrhythmia Detection Models

To train and evaluate full heirarchically gated RSNN, use the following method.

```
python main.py \
--config_file=hgrsnn_config.yaml \
--model_name=hgrsnn \
--data_dir={path to processed data} \
--output_dir={output path} \
--num_seeds=5 --train --use_best_model
```