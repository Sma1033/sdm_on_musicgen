# Detecting Training Data in Music Generation Models with Sample Distance Matrix (SDM)

This repository contains the supplementary code and data for the paper:

**Detecting Training Data in Music Generation Models with Sample Distance Matrix**  
I-Chieh Wei, Michael Witbrock, Fabio Morreale  
*ICASSP 2026 (submitted)*

---

## 📌 Overview

This project investigates **Membership Inference Attacks (MIA)** on **MusicGen** models.  
We evaluate four baseline MIA methods (perplexity, Min-K% probability, perturbation, fusion) and introduce a novel approach called **Sample Distance Matrix (SDM)**.  

Our results show:
- Existing MIA methods from text transfer poorly to music generation.
- SDM consistently outperforms baselines across multiple genres.
- We release metadata, processed features, and code to enable reproducibility.

---

## 📂 Repository Structure

```
MIA_on_musicgen/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/                     # Core scripts
│   ├── run_musicgen.py       # Run MusicGen on audio → tokens/logits
│   ├── get_tokens.py         # Extract EnCodec tokens
│   ├── get_logits.py         # Extract logits from MusicGen
│   ├── process_results.py    # Convert logits to features for MIA
│   ├── mia_methods.py        # Baseline + SDM MIA implementations
│   └── evaluate_auc.py       # AUC/ROC evaluation
│
└── data/
    ├── metadata/             # Song metadata (no raw audio)
    │   ├── sdm_dataset.csv
    │   └── supplementary_dataset.csv
    ├── processed/            # Processed features for reproducibility
    │   ├── logits/*.npy
    │   ├── tokens/*.npy
    │   ├── distances/*.npy
    │   └── auc_results.csv
    └── figures/              # Plots used in the paper
 

```

---

## ⚙️ Installation

We recommend Python 3.10+ and a GPU-enabled environment (CUDA).

```bash
git clone https://github.com/yourname/MIA_on_musicgen.git
cd MIA_on_musicgen
pip install -r requirements.txt
```

### Dependencies
- [audiocraft (MusicGen)](https://github.com/facebookresearch/audiocraft)
- [encodec](https://github.com/facebookresearch/encodec)
- [demucs](https://github.com/facebookresearch/demucs) (for vocal removal)
- numpy, torch, torchaudio, scikit-learn, matplotlib

---

## 🚀 Usage

### 1. Run MusicGen and extract logits/tokens
```bash
python src/run_musicgen.py --input data/raw/ --output data/processed/logits --model medium
```

This will save:
- `xxx_tokens.npy` (EnCodec tokens)
- `xxx_logits.npy` (per-timestep logits)

### 2. Process results into features
```bash
python src/process_results.py --input data/processed/logits --output data/processed/distances
```

### 3. Run MIA methods
```bash
python src/mia_methods.py --input data/processed/distances --method SDM
```

### 4. Evaluate with AUC-ROC
```bash
python src/evaluate_auc.py --input results/sdm_predictions.npy
```

---

## 📊 Data Availability

- **No raw audio is included** in this repository (due to copyright).  
- We provide instead:
  - **Metadata CSVs** listing member / non-member songs.
  - **Processed features** (`tokens`, `logits`, `distance matrices`).
  - **AUC results** as CSV for reproducibility.

If you are interested in re-running experiments with the same audio, please obtain the original songs legally (see `data/metadata/` for sources).

---

## 📈 Results (from the paper)

| Method                  | Overall AUC |
|--------------------------|-------------|
| Raw perplexity           | 0.578       |
| Min-K% Probability (10%) | 0.591       |
| Perturbation             | 0.595       |
| Fusion                   | 0.529       |
| **Sample Distance Matrix (SDM)** | **0.666** |
| **SDM + Supplementary Dataset** | **0.707** |



---

## 📜 Ethical Statement

This project used only publicly available datasets under their respective licenses.  
No human subjects or personal data were involved.  
All experiments were conducted solely for academic research with no commercial use.

---

## ✨ Acknowledgements

- [Meta AI](https://github.com/facebookresearch/audiocraft) for MusicGen and EnCodec.
- [Demucs](https://github.com/facebookresearch/demucs) for vocal separation.
- The University of Auckland.
