import os
import pandas as pd

#  path
base_dir = '/Users/ling/mir_datasets/medley_solos_db'
annotation_csv = os.path.join(base_dir, 'Medley-solos-DB_metadata.csv')
#  original path 
selected_audio_dir = os.path.join(base_dir, 'audio')                      
output_csv = os.path.join(base_dir, 'audio_labels_1000.csv')              # output csv

# check the path
assert os.path.exists(annotation_csv), f"can't find original label: {annotation_csv}"
assert os.path.exists(selected_audio_dir), f"can't find catalogue: {selected_audio_dir}"

# read original metadata
meta_df = pd.read_csv(annotation_csv)

# get file name
selected_files = [f for f in os.listdir(selected_audio_dir) if f.endswith('.wav')]

selected_df = meta_df[meta_df['filename'].isin(selected_files)]

# new csv
selected_df.to_csv(output_csv, index=False)
