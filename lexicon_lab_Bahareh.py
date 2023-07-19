#!/usr/bin/env python
# coding: utf-8

# In[15]:


#!pip install gensim numpy pandas matplotlib


# In[16]:


#: Import required libraries.
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos_sim

import matplotlib.pyplot as plt
from switch import switch_simdrop



# In[17]:


# Function to load embeddings
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings



# Load model_vectors data
word2vec_vectors = load_embeddings('word2vec.txt')
speech2vec_vectors = load_embeddings('speech2vec.txt')



# Load participant data
data = pd.read_csv('data-cochlear.txt', sep='\t', names=['Participant', 'Word'])
data


# In[24]:


from sklearn.metrics.pairwise import cosine_similarity


class Similarity:
    def __init__(self):
        pass
    
    def cosine_similarity_func(self, word1, word2, embeddings):
        vector1, vector2 = None, None
        
        if word1 in embeddings and word2 in embeddings:
            vector1 = embeddings[word1]
            vector2 = embeddings[word2]

        if vector1 is None or vector2 is None:
            return 0

        return cosine_similarity([vector1], [vector2])[0][0]


    def get_vector(self, model, word, default_size=50):  # set default vector size to 50
        if word in model:
            vec = np.array(model[word]).reshape(1, -1)

            return vec, vec.shape[1]
        else:
            return np.zeros((1, default_size)), default_size

    def pairwise_similarity(self, data, model, model_name):
        # Calculate the pairwise similarity between each consecutive word produced by participants based on the provided model
        unique_ids = data['ID'].unique()
        results = []

        for uid in unique_ids:
            current_word = data.loc[data['ID'] == uid, 'Word'].iloc[0]
            similarities = []
            for word in data.loc[data['ID'] == uid, 'Word']:
                if word != current_word:
                    vec1, _ = self.get_vector(model, current_word)
                    vec2, _ = self.get_vector(model, word)
                    similarity = self.cosine_similarity_func(current_word, word, model)
                    similarities.append(float(similarity))
                else:
                    similarities.append(2)
            results.append((uid, word, similarities, model_name))

        pairwise_sim_df = pd.DataFrame(results, columns=['ID', 'Word', 'Similarities', 'Model'])
        return pairwise_sim_df


    
    


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import os

class Clusters:
    def __init__(self):
        self.cluster_data = {}

    def compute_clusters(self, data, word2vec, speech2vec):
        cluster_data = {}
        
        for model_name, model in [("word2vec", word2vec), ("speech2vec", speech2vec)]:
            similarity = Similarity()
            sim_data = similarity.pairwise_similarity(data, model, model_name)
            sim_scores = [item for sublist in sim_data['Similarities'].tolist() for item in sublist]  # Flatten the similarities list
            switches = switch_simdrop(data['Word'].tolist(), sim_scores)
            cluster_data[model_name] = switches
            
        self.cluster_data = cluster_data  # Assigning results to the class attribute
        
    def visualize_clusters(self, save_path="results/cluster_comparison.png"):
        # Calculate means for both models
        means = {}
        for model_name, results in self.cluster_data.items():
            cluster_values = [result for result in results if result != 2]
            switch_values = [result for result in results if result == 1]
            means[model_name] = {
                "Clusters": np.mean(cluster_values) if cluster_values else 0,
                "Switches": np.mean(switch_values) if switch_values else 0
            }

        # Extract means for plotting
        word2vec_means = [means['word2vec']['Clusters'], means['word2vec']['Switches']]
        speech2vec_means = [means['speech2vec']['Clusters'], means['speech2vec']['Switches']]

        # Create the bar plot
        barWidth = 0.3
        r1 = np.arange(len(word2vec_means))
        r2 = [x + barWidth for x in r1]

        plt.bar(r1, word2vec_means, color='b', width=barWidth, edgecolor='grey', label='word2vec')
        plt.bar(r2, speech2vec_means, color='r', width=barWidth, edgecolor='grey', label='speech2vec')
        
        plt.xlabel('Metrics', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(word2vec_means))], ['Clusters', 'Switches'])
        plt.legend()

        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig(save_path)
        plt.show()


