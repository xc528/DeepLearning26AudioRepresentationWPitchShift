import os
import pandas as pd

# path
base_path = '/Users/ling/Downloads/Homework/Deep_Learning/final'
csv_original_path = os.path.join(base_path, 'audio_labels_1000_balanced.csv')

# pitch and sampling rate
SHIFTS = [i for i in range(-5, 6) if i != 0]
SAMPLE_RATES = [24000, 48000]

# read original csv
df_original = pd.read_csv(csv_original_path)
df_original['uuid4'] = df_original['uuid4'].astype(str).str.strip()

# collect all data
all_rows = []

for sr in SAMPLE_RATES:
    for shift in SHIFTS:

        folder_name = f'audio_1000_balanced_pitch_{shift}_{sr//1000}k'
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.exists(folder_path):
            print(f" Missing folder: {folder_name}")
            continue

        audio_files = [f for f in os.listdir(folder_path)
                       if f.endswith('.wav') and not f.startswith('._')]

        for f in audio_files:
            uuid = os.path.splitext(f)[0].split('_')[-1]

            # find label
            row = df_original[df_original['uuid4'] == uuid]

            if len(row) == 0:
                continue

            row_dict = row.iloc[0].to_dict()

            # add information to label
            row_dict['pitch_shift'] = shift
            row_dict['sampling_rate'] = sr
            row_dict['file_path'] = os.path.join(folder_path, f)

            all_rows.append(row_dict)

# output
df_all = pd.DataFrame(all_rows)

output_path = os.path.join(base_path, 'master_metadata.csv')
df_all.to_csv(output_path, index=False)

