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

## Model










## Analysis




