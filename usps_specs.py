#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import tensorflow as tf
from utils import read_list
from sklearn.datasets.mldata import fetch_mldata

# Fetch the dataset
dataset = fetch_mldata("USPS")
print("Dataset USPS loaded...")
data = dataset.data
target = dataset.target - 1 # Labels between 0 and 9 to match digits
n_samples = data.shape[0] # Number of samples in the dataset
n_clusters = 10 # Number of clusters to obtain

# Get the split between training/test set and validation set
test_indices = read_list("split/usps/test")
validation_indices = read_list("split/usps/validation")

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