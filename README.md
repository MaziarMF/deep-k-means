# deep-k-means
Here we have provided the implementation of the "Deep k-Means: Jointly Clustering with k-Means and Learning Representations" based on Python and Tensorflow. The paper is available in following link:
https://arxiv.org/abs/1806.10069

Abstract:"We study in this paper the problem of jointly clustering and learning representations. As several previous studies have shown, learning representations that are both faithful to the data to be clustered and adapted to the clustering algorithm can lead to better clustering performance, all the more so that the two tasks are performed jointly. We propose here such an approach for k-Means clustering based on a continuous reparametrization of the objective function that leads to a truly joint solution. "

# Discription of the files

We have tested our algorithm on 4 datasets: 1-20news group 2-rcv1 3-mnist and 4-usps. For each dataset we have provided a spec file which includes: 
    1-loading of the dataset
    2-any data preprocessing step
    3-values of the hyperparameters
    4-etc.
The utils file includes some basic functions to compute cluster accuracy, shuffling the data and etc.
The computational graph of our proposed deep k-means and DCN paper is provided in compgraph.py. Please note that we reimplemented the DCN paper ("Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering" to compare farely with our approach). 

The files aekm.y and km.py are provided sequentially to compute the results for autoencoder+kmeans and kmeans.

Please fill free to notify us if you found any bugs in the code.

Please cite us in case of using our provided implementation.
