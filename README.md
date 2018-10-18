**Deep k-Means: Jointly Clustering with k-Means and Learning Representations**
======

## __Introduction__

This repository provides the source code for the models and baselines described in *Deep k-Means: Jointly Clustering with k-Means and Learning Representations* by Maziar Moradi Fard, Thibaut Thonet, and Eric Gaussier. The implementation is based on Python and Tensorflow. More details about this work can be found in the original paper, which is available at https://arxiv.org/abs/1806.10069.

**Abstract:** We study in this paper the problem of jointly clustering and learning representations. As several previous studies have shown, learning representations that are both faithful to the data to be clustered and adapted to the clustering algorithm can lead to better clustering performance, all the more so that the two tasks are performed jointly. We propose here such an approach for k-Means clustering based on a continuous reparametrization of the objective function that leads to a truly joint solution.

If you found this implementation useful, please consider citing us:
Moradi Fard, M., Thonet, T., & Gaussier, E. (2018). **[Deep k-Means: Jointly Clustering with k-Means and Learning Representations](https://arxiv.org/abs/1806.10069)**. ArXiv:1806.10069.

Feel free to contact us if you discover any bugs in the code or if you have any questions.

## __Content__

The repository contains the following files:
* The python scripts used to run the different models and baselines: **dkm.py**, **dcn.py**, **aekm.py**, **km.py**. 
* The python script **compgraph.py** containing the Tensorflow computation graph of the different models.
* The python scripts describing the specifications of the datasets (used for dataset loading, preprocessing, and dataset-specific parameter setting): **_20news_specs.py**, **mnist_specs.py**, **rcv1_specs.py**, **usps_specs.py**, which respectively correspond to the datasets 20NEWS, MNIST, RCV1, and USPS.
* The python script **utils.py**, which defines basic functions.
* The directory **split** containing the dataset split used to define validation sets and test sets for the different datasets. Note that the notion of validation set and test set is only used during the evaluation; algorithms are otherwise trained on the whole dataset.
* The file **LICENCE.txt** describing the licence of our code.
* The file **README.md**, which is the current file.

## __How to run the code__

### __Deep k-Means__

Deep k-Means (DKM) is run using the following command:
```dkm.py [-h] -d <string> [-v] [-p] [-a] [-s] [-c] [-l <double>] [-e <int>] [-f <int>] [-b <int>]```

The meaning of each argument is detailed below:
* ``-h``, ``--help``: Show usage.
* ``-d <string>``, ``--dataset <string>``: Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1).
* ``-v``, ``--validation``: Split data into validation and test sets.
* ``-p``, ``--pretrain``: Pretrain the autoencoder and cluster representatives.
* ``-a``, ``--annealing``: Use an annealing scheme for the values of alpha (otherwise a constant is used).
* ``-s``, ``--seeded``: Use a fixed seed, different for each run.
* ``-c``, ``--cpu``: Force the program to run on CPU.
* ``-l <double>``, ``--lambda <double>``: Value of the hyperparameter weighing the clustering loss against the reconstruction loss. Default value: 1.0.
* ``-e <int>``, ``--p_epochs <int>``: Number of pretraining epochs. Default value: 50.
* ``-f <int>``, ``--f_epochs <int>``: Number of fine-tuning epochs per alpha value. Default value: 5.
* ``-b <int>``, ``--batch_size <int>``: Size of the minibatches used by the optimizer. Default value: 256.

**Example:**
Running DKMp (with pretraining and without annealing) on USPS: ```dkm.py -d USPS -v -p -s -l 1.0```

### __Deep Clustering Network__

Note that to facilitate a fair comparison with our approach, we reimplemented in Tensorflow the Deep Clustering Network (DCN) model which was originally proposed in:
Yang, B., Fu, X., Sidiropoulos, N. D., & Hong, M. (2017). **[Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering](https://arxiv.org/abs/1610.04794)**. In Proceedings of the 34th International Conference on Machine Learning (pp. 3861–3870).

The baseline DCN is run using the following command:
```dcn.py [-h] -d DATASET [-v] [-p] [-s] [-c] [-l <double>] [-e <int>] [-f <int>] [-b <int>]```

The meaning of each argument is detailed below:
* ``-h``, ``--help``: Show usage.
* ``-d <string>``, ``--dataset <string>``: Dataset on which DCN will be run (one of USPS, MNIST, 20NEWS, RCV1).
* ``-v``, ``--validation``: Split data into validation and test sets.
* ``-p``, ``--pretrain``: Pretrain the autoencoder and cluster representatives.
* ``-s``, ``--seeded``: Use a fixed seed, different for each run.
* ``-c``, ``--cpu``: Force the program to run on CPU.
* ``-l <double>``, ``--lambda <double>``: Value of the hyperparameter weighing the clustering loss against the reconstruction loss. Default value: 1.0.
* ``-e <int>``, ``--p_epochs <int>``: Number of pretraining epochs. Default value: 50.
* ``-f <int>``, ``--f_epochs <int>``: Number of fine-tuning epochs. Default value: 50.
* ``-b <int>``, ``--batch_size <int>``: Size of the minibatches used by the optimizer. Default value: 256.

**Example:**
Running the original DCN (with pretraining) on USPS: ```dcn.py -d USPS -v -p -s -l 0.1```

### __Auto-Encoder + k-Means__

The baseline Auto-Encoder + k-Means (AEKM) is run using the following command:
```aekm.py [-h] -d <string> [-v] [-s] [-c] [-e <int>] [-b <int>]```

The meaning of each argument is detailed below:
* ``-h``, ``--help``: Show usage.
* ``-d <string>``, ``--dataset <string>``: Dataset on which DCN will be run (one of USPS, MNIST, 20NEWS, RCV1).
* ``-v``, ``--validation``: Split data into validation and test sets.
* ``-s``, ``--seeded``: Use a fixed seed, different for each run.
* ``-c``, ``--cpu``: Force the program to run on CPU.
* ``-e <int>``, ``--p_epochs <int>``: Number of pretraining epochs. Default value: 50.
* ``-b <int>``, ``--batch_size <int>``: Size of the minibatches used by the optimizer. Default value: 256.

**Example:**
Running AEKM on USPS: ```aekm.py -d USPS -v -s```

### __k-Means__

The baseline k-Means (KM) is run using the following command:
```km.py [-h] -d <string> [-v] [-s]```

The meaning of each argument is detailed below:
* ``-h``, ``--help``: Show usage.
* ``-d <string>``, ``--dataset <string>``: Dataset on which DCN will be run (one of USPS, MNIST, 20NEWS, RCV1).
* ``-v``, ``--validation``: Split data into validation and test sets.
* ``-s``, ``--seeded``: Use a fixed seed, different for each run.

**Example:**
Running KM on USPS: ```km.py -d USPS -v -s```