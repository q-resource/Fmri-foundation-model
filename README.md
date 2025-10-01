Brain-DiT Implementation Code (Quick Start)
1.Train and Test code: uvit_EFT_train.py uvit_EFT_inference.py
2.Cognitive Relevance Map(crs) computing code:crs_compute.py
3.Behavioural Results:behavior_data_for_brain-dit.zip

# 🧠 fMRI Foundation Model

> A foundation model for functional MRI (fMRI), enabling brain state representation learning, cognitive task decoding, and brain–behavior association analysis.

---

## 📌 Introduction

Functional MRI (fMRI) provides rich high-dimensional signals of human brain activity. Building a **foundation model** for fMRI can provide unified representations that benefit a wide range of downstream tasks, such as:

- Cognitive task decoding  
- Behavioral / phenotypic prediction  
- Brain–behavior mapping  
- Functional connectivity analysis  

This repository implements a **BrainDiT-based foundation model** for fMRI data, including **training, inference, connectivity computation, and cognitive relevance mapping**.

---

## ✨ Features

- ✅ **Unified framework** for multiple tasks (e.g., EFT, SST)  
- ✅ **Functional Connectivity (FC)** computation (`computed_fcnew.py`)  
- ✅ **Cognitive Relevance Map (CRS)** generation (`crs_compute.py`)  
- ✅ **Flexible dataset interface** for fMRI (`fmridataset.py`)  
- ✅ **Cosine annealing + warmup learning rate schedule** (`cosine_annealing_warmup.py`)  
- ✅ **Transformer/ViT-based models** with rotary embeddings (`models.py`)  
- ✅ **Training & inference scripts** for EFT & SST (`uvit_*_train.py`, `uvit_*_inference.py`)  
- ✅ **Utility functions** for metrics, visualization, preprocessing (`utils.py`)  

---

## 📂 Repository Structure

```

Fmri-foundation-model/
│
├── computed_fcnew.py       # Functional connectivity computation
├── crs_compute.py          # Cognitive relevance map generation
├── cosine_annealing_warmup.py
├── fmridataset.py          # Dataset loading & preprocessing
├── models.py               # Model architectures
├── sde.py                  # Noise / diffusion process
├── uvit_EFT_train.py       # Training script for EFT task
├── uvit_EFT_inference.py   # Inference script for EFT task
├── uvit_SST_train.py       # Training script for SST task
├── uvit_SST_inference.py   # Inference script for SST task
├── utils.py                # Helper functions
└── requirements.txt        # Python dependencies

````

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/q-resource/Fmri-foundation-model.git
cd Fmri-foundation-model
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Typical requirements include:

* Python ≥ 3.8
* PyTorch
* numpy, scipy, pandas
* scikit-learn
* nibabel, nilearn
* matplotlib, seaborn
* tqdm, einops

---

## 📊 Data & Preprocessing

* Input: fMRI time series (NIfTI or ROI-based signals, e.g., `samples × time × nodes`)
* Auxiliary: Task labels, behavioral/phenotypic data (CSV format)

Preprocessing typically includes:

1. Standard fMRI preprocessing (motion correction, normalization, filtering)
2. ROI parcellation or surface mapping
3. Denoising (head motion, WM/CSF regression, band-pass filtering)
4. FC computation (optional)

---

## 🚀 Usage

### 1. Training (example: EFT task)

```bash
python uvit_EFT_train.py --data_path path/to/train_data --epochs 100 --lr 1e-4
```

### 2. Inference

```bash
python uvit_EFT_inference.py --checkpoint path/to/model.ckpt --data_path path/to/test_data
```

### 3. Functional Connectivity

```bash
python computed_fcnew.py --input path/to/fmri_data --output path/to/fc_results
```

### 4. Cognitive Relevance Map

```bash
python crs_compute.py --model_ckpt path/to/model.ckpt --behavior path/to/behavior.csv
```

---

## 📈 Example Workflow

```python
from fmridataset import FMRIDataset
from models import BrainDiTModel
from utils import compute_metrics

# Load dataset
train_ds = FMRIDataset("data/train_fmri.npy", "data/train_labels.csv")

# Initialize model
model = BrainDiTModel(hidden_dim=256, num_layers=8)

# Train
for epoch in range(50):
    train_one_epoch(model, train_ds)

# Inference
preds = model.inference("data/test_fmri.npy")
metrics = compute_metrics(preds, "data/test_labels.csv")
print(metrics)
```

---

## 📊 Results

| Task                | Dataset         | Metric    | Value |
| ------------------- | --------------- | --------- | ----- |
| EFT decoding        | Example dataset | Accuracy  | 78%   |
| SST decoding        | Example dataset | AUC       | 0.82  |
| Behavior prediction | Example dataset | Pearson r | 0.45  |

*(Replace with your real experimental results.)*

---

## 🤝 Contributing

Contributions are welcome!

* Open an **issue** for bugs / feature requests
* Submit a **pull request** for improvements
* Please follow **PEP8** coding style

---

## 📜 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## 📬 Contact

* Author: [q-resource](https://github.com/q-resource)
* Email: *rqhzai21@m.fudan.edu.cn*

If you use this project in your research, please cite it accordingly.

