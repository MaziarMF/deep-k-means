#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import os
import math
import numpy as np
import argparse
import tensorflow as tf
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from utils import cluster_acc
from utils import next_batch
from compgraph import AeCompGraph

parser = argparse.ArgumentParser(description="Deep k-means algorithm.")
parser.add_argument("-d", "--dataset", type=str.upper,
                    help="Dataset on which AEKM will be run (one of USPS, MNIST, 20NEWS, RCV1)", required=True)
parser.add_argument("-v", "--validation", help="Split data into validation and test sets", action='store_true')
parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true')
parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Size of the minibatches used by the optimizer")
args = parser.parse_args()

# Dataset setting from arguments
if args.dataset == "USPS":
    import usps_specs as specs
elif args.dataset == "MNIST":
    import mnist_specs as specs
elif args.dataset == "20NEWS":
    import _20news_specs as specs
elif args.dataset == "RCV1":
    import rcv1_specs as specs
else:
    parser.error("Unknown dataset!")
    exit()

# Parameter setting from arguments
n_train_epochs = args.epochs
batch_size = args.batch_size # Size of the mini-batches used in the stochastic optimizer
n_batches = int(math.ceil(specs.n_samples / batch_size)) # Number of mini-batches
validation = args.validation # Specify if data should be split into validation and test sets
seeded = args.seeded # Specify if runs are seeded

# Select only the labels which are to be used in the evaluation (disjoint for validation and test)
if validation:
    validation_target = np.asarray([specs.target[i] for i in specs.validation_indices])
    test_target = np.asarray([specs.target[i] for i in specs.test_indices])
else:
    target = specs.target

# Dataset on which the computation graph will be run
data = specs.data

# Hardware specifications
if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Run on CPU instead of GPU if batch_size is small
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
config = tf.ConfigProto(gpu_options=gpu_options)

# Definition of the randomly-drawn (0-10000) seeds to be used for each run
seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

if validation:
    list_validation_acc = []
    list_validation_ari = []
    list_validation_nmi = []
    list_test_acc = []
    list_test_ari = []
    list_test_nmi = []
else:
    list_acc = []
    list_ari = []
    list_nmi = []

n_runs = 10
for run in range(n_runs):
    # Use a fixed seed for this run, as defined in list seeds
    if seeded:
        tf.reset_default_graph()
        tf.set_random_seed(seeds[run])
        np.random.seed(seeds[run])

    print("Run", run)

    # Define the computation graph for DKM
    cg = AeCompGraph([specs.dimensions, specs.activations, specs.names])

    # Run the computation graph
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Initialization
        init = tf.global_variables_initializer()
        sess.run(init)

        print("Starting autoencoder training...")

        # Variables to save pretraining tensor content
        embeddings = np.zeros((specs.n_samples, specs.embedding_size), dtype=float)

        # First, pretrain the autoencoder
        ## Loop over epochs
        for epoch in range(n_train_epochs):
            print("Pretraining step: epoch {}".format(epoch))

            # Loop over the samples
            for _ in range(n_batches):
                # Fetch a random data batch of the specified size
                indices_, data_batch = next_batch(batch_size, data)

                # Run the computation graph until pretrain_op (only on autoencoder) on the data batch
                _, embedding_, ae_loss_ = sess.run((cg.train_op, cg.embedding, cg.loss),
                                                   feed_dict={cg.input: data_batch})

                # Save the embeddings for batch samples
                for j in range(len(indices_)):
                    embeddings[indices_[j], :] = embedding_[j, :]

                #print("ae_loss_:", float(ae_loss_))

        # Second, run k-means++ on the pretrained embeddings
        print("Running k-means on the learned embeddings...")
        kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(embeddings)

        if validation:
            # Split the cluster assignments into validation and test ones
            validation_cluster_assign = np.asarray([kmeans_model.labels_[i] for i in specs.validation_indices])
            test_cluster_assign = np.asarray([kmeans_model.labels_[i] for i in specs.test_indices])

            # Evaluate the clustering validation performance using the ground-truth labels
            validation_acc = cluster_acc(validation_target, validation_cluster_assign)
            print("Validation ACC", validation_acc)
            validation_ari = adjusted_rand_score(validation_target, validation_cluster_assign)
            print("Validation ARI", validation_ari)
            validation_nmi = normalized_mutual_info_score(validation_target, validation_cluster_assign)
            print("Validation NMI", validation_nmi)

            # Evaluate the clustering test performance using the ground-truth labels
            test_acc = cluster_acc(test_target, test_cluster_assign)
            print("Test ACC", test_acc)
            test_ari = adjusted_rand_score(test_target, test_cluster_assign)
            print("Test ARI", test_ari)
            test_nmi = normalized_mutual_info_score(test_target, test_cluster_assign)
            print("Test NMI", test_nmi)
        else:
            # Evaluate the clustering performance using the ground-truth labels
            acc = cluster_acc(target, kmeans_model.labels_)
            print("ACC", acc)
            ari = adjusted_rand_score(target, kmeans_model.labels_)
            print("ARI", ari)
            nmi = normalized_mutual_info_score(target, kmeans_model.labels_)
            print("NMI", nmi)

        # Record the clustering performance for the run
        if validation:
            list_validation_acc.append(validation_acc)
            list_validation_ari.append(validation_ari)
            list_validation_nmi.append(validation_nmi)
            list_test_acc.append(test_acc)
            list_test_ari.append(test_ari)
            list_test_nmi.append(test_nmi)
        else:
            list_acc.append(acc)
            list_ari.append(ari)
            list_nmi.append(nmi)

if validation:
    list_validation_acc = np.array(list_validation_acc)
    print("Average validation ACC: {:.3f} +/- {:.3f}".format(np.mean(list_validation_acc), np.std(list_validation_acc)))
    list_validation_ari = np.array(list_validation_ari)
    print("Average validation ARI: {:.3f} +/- {:.3f}".format(np.mean(list_validation_ari), np.std(list_validation_ari)))
    list_validation_nmi = np.array(list_validation_nmi)
    print("Average validation NMI: {:.3f} +/- {:.3f}".format(np.mean(list_validation_nmi), np.std(list_validation_nmi)))

    list_test_acc = np.array(list_test_acc)
    print("Average test ACC: {:.3f} +/- {:.3f}".format(np.mean(list_test_acc), np.std(list_test_acc)))
    list_test_ari = np.array(list_test_ari)
    print("Average test ARI: {:.3f} +/- {:.3f}".format(np.mean(list_test_ari), np.std(list_test_ari)))
    list_test_nmi = np.array(list_test_nmi)
    print("Average test NMI: {:.3f} +/- {:.3f}".format(np.mean(list_test_nmi), np.std(list_test_nmi)))
else:
    list_acc = np.array(list_acc)
    print("Average ACC: {:.3f} +/- {:.3f}".format(np.mean(list_acc), np.std(list_acc)))
    list_ari = np.array(list_ari)
    print("Average ARI: {:.3f} +/- {:.3f}".format(np.mean(list_ari), np.std(list_ari)))
    list_nmi = np.array(list_nmi)
    print("Average NMI: {:.3f} +/- {:.3f}".format(np.mean(list_nmi), np.std(list_nmi)))