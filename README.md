# 🧠 Parkinson's UPDRS Severity Estimator

> Explainable deep learning model for continuous UPDRS motor score regression from voice biomarkers — moving beyond binary PD classification.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-blueviolet)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![IEEE](https://img.shields.io/badge/Target-IEEE%20Access-00629B?logo=ieee)](https://ieeeaccess.ieee.org/)

---

## 📌 Motivation

Most existing Parkinson's disease (PD) AI research frames the problem as **binary classification** (PD vs. healthy). This ignores a clinically critical question: *how severe is the disease?*

This project addresses that gap by:
- Predicting the **continuous UPDRS motor score** (0–108 scale) from voice features
- Using a **Feature-Attention MLP** that learns which biomarkers matter most
- Providing **SHAP-based explanations** that map directly to known clinical biomarkers (jitter, shimmer, NHR)

---

## 🏗️ Project Structure

```
parkinsons-updrs-estimator/
├── data/                      # Raw and processed datasets
│   └── parkinsons_updrs.csv   # UCI Parkinson's Telemonitoring Dataset
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_baselines.ipynb     # RF, SVR, plain MLP baselines
│   └── 03_main_model.ipynb    # Feature-Attention MLP + SHAP
├── src/
│   ├── dataset.py             # Data loading + subject-wise splitting
│   ├── model.py               # Feature-Attention MLP architecture
│   ├── train.py               # Training loop + evaluation
│   ├── explain.py             # SHAP explanation pipeline
│   └── utils.py               # Metrics, plotting helpers
├── models/                    # Saved model checkpoints
├── results/
│   ├── figures/               # All publication-ready plots
│   └── metrics/               # JSON logs of all experiment results
├── app/
│   └── streamlit_app.py       # Interactive demo
├── docs/
│   └── paper_draft.md         # IEEE paper draft (in progress)
├── requirements.txt
└── README.md
```

---

## 🔬 Novel Contributions

| Aspect | Prior Work | This Work |
|--------|-----------|-----------|
| Output | Binary (PD / Healthy) | Continuous UPDRS score (regression) |
| Architecture | Standard MLP / CNN | Feature-Attention MLP |
| Explainability | Post-hoc SHAP bolt-on | SHAP + attention weights integrated |
| Clinical mapping | Absent | Top features mapped to PD biomarkers |

---

## 🗃️ Dataset

**UCI Parkinson's Telemonitoring Dataset**
- 5,875 voice recordings from 42 subjects
- 22 voice biomarker features (jitter, shimmer, HNR, RPDE, DFA, PPE, etc.)
- Target: `total_UPDRS` score (continuous, 7–54.99 range in this dataset)
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
- Split strategy: **subject-wise** (no data leakage across train/test)

---

## 🧩 Architecture

```
Input (22 voice features)
        │
        ▼
Feature Attention Layer     ← learnable softmax weights per feature
        │
        ▼
FC Layer 1 (256) + BatchNorm + ReLU + Dropout(0.3)
        │
        ▼
FC Layer 2 (128) + BatchNorm + ReLU + Dropout(0.3)
        │
        ▼
FC Layer 3 (64)  + ReLU
        │
        ▼
Output: UPDRS score (scalar regression)
```

---

## 📊 Results (Preliminary)

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|-------|--------|------|
| Random Forest | — | — | — |
| SVR | — | — | — |
| Plain MLP | — | — | — |
| **Feature-Attention MLP (Ours)** | **—** | **—** | **—** |

*Results will be updated as experiments complete.*

---

## ⚡ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Stokesy-dev/parkinsons-updrs-estimator.git
cd parkinsons-updrs-estimator

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip3 install -r requirements.txt

# 4. Download dataset (automated)
python3 src/dataset.py --download

# 5. Run training
python3 src/train.py --model attention_mlp --epochs 100

# 6. Launch demo app
streamlit run app/streamlit_app.py
```

---

## 🔍 Explainability

SHAP values reveal which voice features most strongly predict UPDRS severity:

- **Jitter (%)** — vocal frequency instability, key PD marker
- **Shimmer (dB)** — amplitude variation, linked to motor rigidity  
- **NHR / HNR** — noise-to-harmonics ratio, breath control indicator
- **RPDE / DFA** — nonlinear dynamical complexity measures

*SHAP summary plots and per-patient force plots available in `results/figures/`.*

---

## 🚀 Demo

Run the Streamlit app to input voice features and get:
- Predicted UPDRS severity score
- Severity category (mild / moderate / severe)
- SHAP explanation of which features drove the prediction

```bash
streamlit run app/streamlit_app.py
```

---

## 📄 Paper

This project is being prepared for submission to **IEEE Access** / **IEEE EMBC 2026**.

**Working title:** *Explainable UPDRS Motor Severity Estimation in Parkinson's Disease from Voice Biomarkers using a Feature-Attention Neural Network*

Draft available in `docs/paper_draft.md` (in progress).

---

## 🗓️ Roadmap

- [x] Repository setup + structure
- [ ] EDA notebook
- [ ] Baseline models (RF, SVR, MLP)
- [ ] Feature-Attention MLP implementation
- [ ] SHAP integration
- [ ] Streamlit demo
- [ ] Paper draft
- [ ] IEEE submission

---

## 👤 Author

**Soham** — Final Year BTech (AI & Data Science), MIT-WPU  
GitHub: [@Stokesy-dev](https://github.com/Stokesy-dev)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- UCI ML Repository for the Parkinson's Telemonitoring dataset (A. Tsanas et al., 2010)
- SHAP library by Lundberg & Lee (2017)
