# AlDDoS Classification Pipeline

[![CI Status](https://github.com/yourusername/AlDDoS-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/AlDDoS-Classification/actions)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A deep-learning pipeline for multiclass classification of Application-Layer DDoS (AlDDoS) attacks across five protocols (DNS, LDAP, SNMP, NTP, TFTP) using LSTM, BiLSTM, CNN, and MLP models on the CICDDoS2019 dataset.

---

## 🚀 Features

- **Data Sampling & Merging**  
  Randomly sample 20 000 attack and 4 000 benign records per protocol, then merge into a balanced multiclass CSV.
- **Train/Test Split**  
  Divide the merged dataset into training and testing sets.
- **One-Hot Encoding**  
  Convert all categorical features into binary vectors.
- **Label Remapping**  
  Map tags to multiclass labels (0 = Normal, 1 = DoS, 2 = Probe, 3 = R2L, 4 = U2R).
- **Model Training**  
  Train and compare LSTM, BiLSTM, CNN, and MLP classifiers.
- **Hyperparameter Optimization & Evaluation**  
  Tune model parameters and report accuracy & F1-scores.

---

## 📁 Project Structure

```bash
AlDDoS-Classification/
├── data/
│   ├── raw/                   # Original CSVs per protocol
│   └── sampled/               # Sampled & merged datasets
├── notebooks/
│   ├── 1_sampling.ipynb       # Sampling & merging workflow
│   ├── 2_preprocessing.ipynb  # Encoding & train/test split
│   └── 3_model_training.ipynb # Model training & evaluation
├── scripts/
│   ├── sample_merge.py        # Sample & merge CSV files
│   ├── preprocess.py          # One-hot encode & split data
│   ├── train_model.py         # Train specified model
│   └── evaluate.py            # Evaluate a trained model
├── models/                    # Saved model checkpoints
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```
## 🛠 Installation
### 1. Clone the repo

``` bash
git clone https://github.com/yourusername/AlDDoS-Classification.git
cd AlDDoS-Classification

```
### 2. Create & activate a virtual environment

``` bash
python3 -m venv env
source env/bin/activate

```
### 3. Install dependencies

``` bash
pip install -r requirements.txt

```
## 🚀 Usage
### 1. Sample & Merge
```bash
python scripts/sample_merge.py \
  --input-dir data/raw \
  --output data/sampled/merged.csv \
  --attack-per-protocol 20000 \
  --benign-per-protocol 4000
```
### 2. Preprocess (Encode & Split)
```bash
python scripts/preprocess.py \
  --input data/sampled/merged.csv \
  --train data/sampled/train.csv \
  --test data/sampled/test.csv
```
### 3. Train a Model

```bash
python scripts/train_model.py \
  --model lstm \
  --train data/sampled/train.csv \
  --output models/lstm.pth
Supported models: lstm, bilstm, cnn, mlp

```
### 4. Evaluate a Model
```bash
python scripts/evaluate.py \
  --model-path models/lstm.pth \
  --test data/sampled/test.csv
```
## 📈 Performance

| Model   | Accuracy | F1-Score | Notes                            |
| ------- | -------- | -------- | -------------------------------- |
| LSTM    | > 99%    | > 99%    | Excellent multiclass performance |
| BiLSTM  | > 99%    | > 99%    | Comparable to LSTM               |
| CNN     | > 98%    | > 98%    | Robust across all five protocols |
| MLP     | > 99%    | > 99%    | Fast training, high accuracy     |


## 📄 Abstract
Abstract
Availability is an attribute of computer and network security which guarantees that resources are available to authorized users. Application-Layer DDoS (AlDDoS) attacks mimic legitimate traffic across OSI layers, making them especially challenging to detect. We present a deep-learning classification pipeline—using LSTM, BiLSTM, CNN, and MLP—on the CICDDoS2019 dataset. Three models achieved > 99% accuracy, while CNN exceeded 98% across all class labels.
