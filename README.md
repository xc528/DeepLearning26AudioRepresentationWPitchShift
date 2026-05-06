# DeepLearning26AudioRepresentationWPitchShift
## Data - Qingjuan 
### label1000.py
#### Subset Metadata Extraction
This script extracts metadata for a selected subset of audio files from the Medley-solos-DB dataset and generates a corresponding CSV file.
##### Input:
1. Original Metadata
File: `Medley-solos-DB_metadata.csv`
Contains full annotations for all audio files
2. Selected Audio Directory
Folder: `audio/`
Contains a subset of `.wav` files (e.g., 1000 selected samples)
##### Purpose:
Match selected audio files with their metadata.
Filter the original dataset.
Generate a new CSV file for the selected subset.
##### Output: 
File: `audio_labels_1000.csv`
Contains:
· filename
· instrument label
· other metadata from original dataset

### data-pitchshifting.py
#### Master Metadata Construction
This script generates a unified metadata file (.csv) that integrates all processed audio data across different pitch perturbations and sampling rates.

##### Goal: 
Consolidate all processed audio files into a single metadata table.
Link each audio file to its corresponding label.
Record additional experiment parameters such as:
`pitch shift`
`sampling rate`
`file path`
##### Input:
1. Base Metadata
File: audio_labels_1000_balanced.csv
Contains original labels for the selected 1000 audio samples
2. Processed Audio Folders
Generated from previous steps:
`audio_1000_balanced_pitch_-5_24k/
audio_1000_balanced_pitch_-4_24k/
...
audio_1000_balanced_pitch_5_48k/`
Each folder contains audio files with a specific:
`pitch shift`
`sampling rate`
##### Output:
master_metadata.csv

Each row contains:
1000 samples × 10 pitch shifts × 2 sampling rates = 20000 rows

## Models

For this project, we used two pre-trained audio embedding models: **OpenL3** and **MusicFM**. Both models were used as feature extractors to generate embeddings from the audio dataset, which were later used for downstream classification experiments.

### Workflow

The general workflow for both models was:

1. Load the pre-trained model.
2. Evaluate the model on the original baseline dataset and the augmented dataset.
3. Extract embeddings from each audio file.
4. Save the extracted embeddings as `.npy` files.
5. Save corresponding metadata, such as file name, label, and dataset split, as `.csv` files.

### MusicFM - Rhan

**MusicFM** is a foundation model for music informatics. It is trained using self-supervised learning and is designed to produce meaningful audio representations from music signals without requiring manually labeled training data.

MusicFM follows the BEST-RQ structure, which stands for Self-supervised Learning with Random-projection Quantizer. In this approach, the model learns by predicting masked audio representations using target tokens generated from random projection and quantization. Unlike general audio models, MusicFM is specifically trained on music data, making it suitable for music-related tasks such as instrument classification, music tagging, and audio similarity analysis.

The MusicFM architecture uses a Conformer encoder, which is a convolution-augmented Transformer. This allows the model to capture both local acoustic patterns through convolutional layers and longer-range temporal relationships through Transformer-based attention.

In this project, we used MusicFM-MSD, a version of MusicFM pre-trained on the entire [Million Song Dataset](http://millionsongdataset.com). Before being passed into the model, all input audio was resampled to 24 kHz and converted to mono. MusicFM outputs frame-level embeddings at 25 Hz, where each frame has an embedding dimension of 1024.

Since each audio clip produces a sequence of frame-level embeddings over time, we applied mean pooling across the time dimension. This converts the frame-level output into a single fixed-length embedding vector for each audio file:

```text
frame-level embeddings → mean pooling over time → one 1024-dimensional embedding vector
```
The example usage for MusicFM is included in the repository under:
```text
Models/musicFM/DeepLearningModelMusicFM.ipynb
```

### OpenL3 - Cathy 
**OpenL3** is a deep learning–based audio embedding model that learns meaningful representations of sound by exploiting the natural correspondence between audio and visual signals in video data. By training on large-scale video datasets, the model captures both low-level acoustic features (like timbre and pitch) and higher-level semantic information (such as musical events or environments).

In this project, we use a version of OpenL3 that has been trained on datasets dominated by musical performances, making it particularly well-suited for analyzing music-related audio.

The input to the model consists of mono audio clips resampled to 48 kHz, ensuring consistency with the model’s expected format. The audio is processed in short overlapping time windows, producing a sequence of embeddings over time.

For each time window, the model outputs a 512-dimensional embedding vector, resulting in a time series of feature vectors that describe how the audio content evolves.

To obtain a single representation for an entire audio clip, we apply mean pooling, which averages the embeddings across all time steps. This reduces the sequence of vectors into one fixed-length 512-dimensional vector, effectively summarizing the overall characteristics of the audio signal.

The code for OpenL3 processing can be found here:
```text
Models/OpenL3/DL_OpenL3.ipynb
```

## Analysis 
### Visualization - Leo
Embedding of models: **Music FM** & **Open L3**
1. Different pitch transposition group (-5 to 5 semitones)
2. torch.Size([1000, 1024])
1000 Validation Samples (Each .pt file represents one sample )
1024 Vector Dimensions per sample

All analysis lives in [analysis/Embedding_analysis.ipynb](analysis/Embedding_analysis.ipynb).

### Step 1 — UMAP Visualization (`embedding_preprocess.py`)

Each model's embeddings are projected from high-dimensional space down to 2D using **UMAP** with cosine distance, then plotted as interactive Plotly scatter plots colored by instrument class.

```
UMAP settings: n_neighbors=15, min_dist=0.1, metric='cosine'
```

This lets us visually inspect whether the embedding space clusters instruments coherently and how cluster structure shifts under pitch perturbation.

**Instrument classes (8):** Clarinet · Guitar · Female Vocals · Flute · Piano · Saxophone · Trumpet · Violin

---

### Step 2 — MLP Classifier Training (`classifier.py`)

A lightweight **MLP classifier** is trained on top of the frozen embeddings to quantify how much instrument information is retained at each pitch shift.

**Classifier architecture:**

```
Input (1024 or 512)
  → Linear(→1024) + BatchNorm + ReLU + Dropout(0.3)
  → Linear(→512)  + BatchNorm + ReLU + Dropout(0.3)
  → Linear(→256)  + BatchNorm + ReLU + Dropout(0.3)
  → Linear(→8)    [output: 8 instrument classes]
```

**Training setup:**

| Setting | Value |
|---|---|
| Split | 70% train / 15% val / 15% test (seed=42) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (on val loss) |
| Epochs | 100 (best val-accuracy checkpoint saved) |
| Loss | CrossEntropyLoss |

---

### Step 3 — Confusion Matrices (`confusion_matrix.py`)

Per-class confusion matrices are generated for every (model, pitch-shift) combination and saved to [analysis/img/](analysis/img/). These reveal which instruments are most susceptible to mis-classification when pitch is altered.

## Analysis
### Quantitative Comparison - Yian

Aggregates Leo's per-shift outputs across 5 shift levels (−5, −1, 0, +1, +5) × 2 models = **10 (model, shift) configurations × 8 instruments**, to characterize how pitch perturbation affects classification structure beyond what any single shift level can show.


### Step 1 — Accuracy curve aggregation

Test accuracies extracted from each of the 10 confusion matrices, plotted as a function of pitch shift (one curve per model).

| Shift | OpenL3 | MusicFM | Gap |
|---|---|---|---|
| −5 | 99.3% | 96.0% | 3.3 |
| −1 | 98.0% | 95.3% | 2.7 |
| 0 | 99.3% | 96.7% | 2.6 |
| +1 | 97.3% | 95.3% | 2.0 |
| +5 | 99.3% | 97.3% | 2.0 |

Both curves stay flat across shifts. OpenL3 leads MusicFM by **2–3 percentage points at every shift level** — no degradation at extremes for either model.

### Step 2 — Confusion structure analysis

Off-diagonal entries catalogued and compared against the baseline (shift = 0).

**Persistent confusions** (baseline + every shifted version):
- **MusicFM:** Clarinet ↔ Flute, Clarinet → Trumpet, Guitar → Clarinet / Piano
- **OpenL3:** Guitar → Piano (every shift)

**Emergent confusions** (shift-only): **none.** Every error pattern observed at ±5 already exists at shift = 0.

→ Pitch perturbation amplifies existing weaknesses rather than introducing new failure modes.

### Step 3 — Per-instrument fragility

Per-class F1 stratified across shifts to identify systematically fragile vs robust classes.

| Class | OpenL3 (+5 semi) | MusicFM (−5 semi) |
|---|---|---|
| Clarinet | 0.97 | 0.89 |
| Flute | 0.97 | 0.92 |
| Piano | 1.00 | 0.93 |
| Guitar | 1.00 | 0.95 |
| Saxophone | 1.00 | 0.96 |
| Trumpet | 1.00 | 1.00 |
| Violin | 1.00 | 1.00 |
| Female Vocals | 1.00 | 1.00 |

**Most fragile:** Clarinet, Flute — sustained-tone wind instruments whose timbre is tightly coupled to the harmonic series.
**Most robust:** Female Vocals (formant-locked), Saxophone (reed signature), Violin (bowed attack) — distinguishing features sit outside the shifted fundamental.

### Methodological caveat

Within-shift training measures **discriminability preservation**, not **pitch invariance**. A pitch-sensitive embedding could still produce flat accuracy if classes remain separable in the shifted space.



