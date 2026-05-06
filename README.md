# DeepLearning26AudioRepresentationWPitchShift
## Data
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

### MusicFM

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
## Analysis




