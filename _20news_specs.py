#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import tensorflow as tf
from utils import read_list
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch the dataset
_20news = fetch_20newsgroups(subset="all")
print("Dataset 20NEWS loaded...")
data = _20news.data
target = _20news.target

# Pre-process the dataset
vectorizer = TfidfVectorizer(max_features=2000)
data = vectorizer.fit_transform(data) # Keep only the 2000 top words in the vocabulary
data = data.toarray() # Switch from sparse matrix to full matrix
n_samples = data.shape[0] # Number of samples in the dataset
n_clusters = 20 # Number of clusters to obtain

# Get the split between training/test set and validation set
test_indices = read_list("split/20news/test")
validation_indices = read_list("split/20news/validation")

# Auto-encoder architecture
input_size = data.shape[1]
hidden_1_size = 500
hidden_2_size = 500
hidden_3_size = 2000
embedding_size = n_clusters
dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names