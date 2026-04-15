# DeepLearning26AudioRepresentationWPitchShift
## Data
-----------------
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
##### How it works?
Step 1: Load Original Metadata
meta_df = pd.read_csv(annotation_csv)
Step 2: Get Selected Audio Files
selected_files = [f for f in os.listdir(selected_audio_dir) if f.endswith('.wav')]
Step 3: Filter Metadata
selected_df = meta_df[meta_df['filename'].isin(selected_files)]
Step 4: Save New CSV
selected_df.to_csv(output_csv, index=False)
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










## Model











## Analysis




