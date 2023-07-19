#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
This notebook aims to compare the performance of two semantic models: word2vec and speech2vec.
We will use the Static Foraging Model and the previous data and models to measure the performance.
"""



# In[1]:


# Necessary imports
import numpy as np
import pandas as pd
import argparse
from scipy.optimize import fmin
from foraging import forage
from cues import create_history_variables
import pandas as pd
import numpy as np
from scipy.optimize import fmin
import os, sys
from tqdm import tqdm
import lexicon_lab_Bahareh
from lexicon_lab_Bahareh import Similarity, Clusters
from foraging import forage
from collections import Counter
import csv



# In[3]:


# Load similarity data
cosine_similarity_data = pd.read_csv('results/cosine_similarity_results.csv')
pairwise_similarity_data = pd.read_csv('results/pairwise_similarity_results.csv')

# Split the data based on model
word2vec_cosine_similarities = cosine_similarity_data[cosine_similarity_data['Model'] == 'word2vec']['Similarity'].values
speech2vec_cosine_similarities = cosine_similarity_data[cosine_similarity_data['Model'] == 'speech2vec']['Similarity'].values

word2vec_pairwise_similarities = pairwise_similarity_data[pairwise_similarity_data['Model'] == 'word2vec']['Similarity'].values
speech2vec_pairwise_similarities = pairwise_similarity_data[pairwise_similarity_data['Model'] == 'speech2vec']['Similarity'].values

# Frequency data loading code

def get_frequencies(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        writer = csv.writer(f_out)
        for line in f_in:
            word, *vector = line.strip().split(' ')
            vector = list(map(float, vector))
            counts = Counter(i for i, value in enumerate(vector) if abs(value) > 0.5)
            for index, count in counts.items():
                writer.writerow([word, index, count])

get_frequencies('word2vec.txt', 'word2vec_frequencies.csv')
get_frequencies('speech2vec.txt', 'speech2vec_frequencies.csv')


# Load word2vec frequencies
word2vec_frequencies_df = pd.read_csv('word2vec_frequencies.csv', names=['word', 'index', 'count'])

# Load speech2vec frequencies
speech2vec_frequencies_df = pd.read_csv('speech2vec_frequencies.csv', names=['word', 'index', 'count'])

# Load frequency data into dictionaries
word2vec_frequencies_dict = word2vec_frequencies_df.groupby('word')['count'].sum().to_dict()
speech2vec_frequencies_dict = speech2vec_frequencies_df.groupby('word')['count'].sum().to_dict()

# Load cosine similarity data
cosine_data = pd.read_csv('results/cosine_similarity_results.csv')



# In[4]:


# Assuming that 'word' in the pairwise_similarity_data is representative of the words
# in the experiment, you can extract the frequencies for the words in the order they appear:
word2vec_frequencies = [word2vec_frequencies_dict.get(word, 0) for word in pairwise_similarity_data[pairwise_similarity_data['Model'] == 'word2vec']['Word']]
speech2vec_frequencies = [speech2vec_frequencies_dict.get(word, 0) for word in pairwise_similarity_data[pairwise_similarity_data['Model'] == 'speech2vec']['Word']]


# In[5]:


beta = (0.5, 0.5)  # Both beta_frequency and beta_semantic set to 0.5 as an example




# In[6]:


#modify the model static
def model_static(beta, freql, freqh, siml, simh):
    
    ct = 0
    all_zero_denrat = True
    for k in range(0, len(freql)):
        if k == 0:
            numrat = pow(freql[k], beta[0])
            denrat = sum(pow(freqh[i], beta[0]) for i in range(k+1))
        else:    
            numrat = pow(freql[k], beta[0]) * pow(siml[k], beta[1])
            denrat = sum(pow(freqh[i], beta[0]) * pow(simh[i], beta[1]) for i in range(k+1))
        
        # Check for denrat being zero and handle it
        if denrat == 0:
            print(f"denrat is zero at index {k}!")
            prob = 0
        else:
            all_zero_denrat = False
            prob = numrat/denrat
        
        ct -= np.log(prob + 1e-10)  # Small constant added to prevent log(0)

    if all_zero_denrat:
        return float('inf')  # Extremely large negative log-likelihood

    return ct

   


# In[7]:


nll_word2vec = model_static(beta, word2vec_frequencies, word2vec_frequencies, word2vec_pairwise_similarities, word2vec_pairwise_similarities)
nll_speech2vec = model_static(beta, speech2vec_frequencies, speech2vec_frequencies, speech2vec_pairwise_similarities, speech2vec_pairwise_similarities)

print(f"Negative Log-Likelihood for word2vec: {nll_word2vec}")
print(f"Negative Log-Likelihood for speech2vec: {nll_speech2vec}")

# The lower value indicates the better model:
if nll_word2vec < nll_speech2vec:
    print("word2vec is a better model for the data.")
else:
    print("speech2vec is a better model for the data.")


# In[18]:


"""
Negative Log-Likelihood for word2vec: 16446.627511091818
Negative Log-Likelihood for speech2vec: 15738.226929820421
speech2vec is a better model for the data.
"""

