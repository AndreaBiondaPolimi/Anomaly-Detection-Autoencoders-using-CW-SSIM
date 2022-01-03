print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

import tensorflow as tf


def color_quantization(img_batch):
    n_colors = 20

    for china in img_batch:
        # Load Image and transform to a 2D numpy array.
        w, h, d = tuple(china.shape)
        assert d == 3
        image_array = tf.reshape(china, (w * h, d))

        def input_fn():
            return tf.data.Dataset.from_tensors(image_array).repeat(1)

        num_clusters = 20
        kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)

        num_iterations = 10
        #previous_centers = None
        for _ in range(num_iterations):
            kmeans.train(input_fn)
            cluster_centers = kmeans.cluster_centers()        
        
        # map the input points to their clusters
        cluster_indices = list(kmeans.predict_cluster_index(input_fn))



        def recreate_image(codebook, labels, w, h):
            """Recreate the (compressed) image from the code book & labels"""
            return codebook[labels].reshape(w, h, -1)
    
        plt.imshow(recreate_image(cluster_centers, cluster_indices, w, h))
        plt.show()

    


       


def sklearn_color_quantization(img_batch):
    n_colors = 20

    for china in img_batch:
        # Load Image and transform to a 2D numpy array.
        w, h, d = tuple(china.shape)
        assert d == 3
        image_array = np.reshape(china, (w * h, d))

        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        print(f"done in {time() - t0:0.3f}s.")

        # Get labels for all points
        print("Predicting color indices on the full image (k-means)")
        t0 = time()
        labels = kmeans.predict(image_array)
        print(f"done in {time() - t0:0.3f}s.")

        codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
        print("Predicting color indices on the full image (random)")

        labels_random = pairwise_distances_argmin(codebook_random,image_array,axis=0)



        def recreate_image(codebook, labels, w, h):
            """Recreate the (compressed) image from the code book & labels"""
            return codebook[labels].reshape(w, h, -1)


        # Display all results, alongside original image
        plt.figure(1)
        plt.clf()
        plt.axis('off')
        plt.title('Original image (96,615 colors)')
        plt.imshow(china)

        plt.figure(2)
        plt.clf()
        plt.axis('off')
        plt.title(f'Quantized image ({n_colors} colors, K-Means)')
        plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

        plt.figure(3)
        plt.clf()
        plt.axis('off')
        plt.title(f'Quantized image ({n_colors} colors, Random)')
        plt.imshow(recreate_image(codebook_random, labels_random, w, h))
        plt.show()