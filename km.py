#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import numpy as np
import argparse
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from utils import cluster_acc
from utils import shuffle

parser = argparse.ArgumentParser(description="k-means algorithm")
parser.add_argument("-d", "--dataset", type=str.upper,
                    help="Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1)", required=True)
parser.add_argument("-v", "--validation", help="Split data into validation and test sets", action='store_true')
parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
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
validation = args.validation # Specify if data should be split into validation and test sets
seeded = args.seeded # Specify if runs are seeded

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
    np.random.seed(seeds[run])

    # Shuffle the dataset
    data, target, indices = shuffle(specs.data, specs.target)

    print("Run", run)
    # Run k-means(++) on the original data
    print("Running k-means on the original data...")
    kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(data)

    if validation:
        validation_indices = [np.where(specs.validation_indices[i] == indices)[0][0]
                              for i in range(len(specs.validation_indices))]
        test_indices = [np.where(specs.test_indices[i] == indices)[0][0]
                        for i in range(len(specs.test_indices))]

        # Select only the labels which are to be used in the evaluation (disjoint for validation and test)
        validation_target = np.asarray([target[i] for i in validation_indices])
        test_target = np.asarray([target[i] for i in test_indices])

        # Split the cluster assignments for validation and test sets
        validation_cluster_assign = np.asarray([kmeans_model.labels_[i] for i in validation_indices])
        test_cluster_assign = np.asarray([kmeans_model.labels_[i] for i in test_indices])

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

        list_validation_acc.append(validation_acc)
        list_validation_ari.append(validation_ari)
        list_validation_nmi.append(validation_nmi)
        list_test_acc.append(test_acc)
        list_test_ari.append(test_ari)
        list_test_nmi.append(test_nmi)
    else:
        acc = cluster_acc(target, kmeans_model.labels_)
        print("ACC", acc)
        ari = adjusted_rand_score(target, kmeans_model.labels_)
        print("ARI", ari)
        nmi = normalized_mutual_info_score(target, kmeans_model.labels_)
        print("NMI", nmi)

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