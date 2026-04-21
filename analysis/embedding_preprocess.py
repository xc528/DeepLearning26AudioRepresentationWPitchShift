import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import torch

class EmbeddingPreprocessor:
    def __init__(self, embeddings_dir, model_name):
        self.embeddings_dir = embeddings_dir
        self.model_name = model_name
        self.instrument_classes = { '0' :'Clarinet', '1':'Guitar', '2':'Female Vocals', '3':'Flute', '4':'Piano', '5':'Saxophone', '6':'Trumpet', '7':'Violin' }
        self.reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    
    def visualize_embeddings(self):
    
        embedding, file_id, instrument_id = self.preprocess_embeddings()
        embedding_2d = self.reducer.fit_transform(embedding)
        instrument_labels = [self.instrument_classes[str(i)] for i in instrument_id]

        #dataframe
        df_projection = pd.DataFrame(embedding_2d, columns=['X', 'Y'])
        df_projection['file_id'] = file_id
        df_projection['instrument_id'] = instrument_labels
        df_projection.head()

        folder_name = os.path.basename(self.embeddings_dir)
        model = self.model_name
        title = f'UMAP Projection of {model} Embeddings {folder_name} semitone'

        fig = px.scatter(df_projection, x='X', y='Y', color='instrument_id', hover_data= 'file_id', title=title, template='plotly_dark')
        fig.show()


    def preprocess_embeddings(self):
        all_embeddings = []
        all_id = []
        all_instrument_id = []

        #add all embeddings from the directory to a list
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith('.pt'):
                file_path = os.path.join(self.embeddings_dir, filename)
                embeddings = torch.load(file_path)
                item_embedding = embeddings.get('embedding')
                id_embedding = embeddings.get('source_file')
                instrument_id = embeddings.get('instrument_id')

                if item_embedding is None:
                    print(f"Warning: File {filename} does not contain the 'embedding' key! Skipping this file.")
                    continue

                all_embeddings.append(item_embedding)
                all_id.append(id_embedding)
                all_instrument_id.append(instrument_id)

        # Concatenate all embeddings into a single array
        all_embeddings = torch.stack(all_embeddings)

        return all_embeddings, all_id, all_instrument_id

