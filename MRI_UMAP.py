# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nibabel as nib
import umap
import sklearn.cluster as cluster
from scipy import ndimage



#%%

img_input = nib.load(r'C:/Users/rallapallih2/Desktop/test/testimg_downsampled.nii.gz')
img_shape = img_input.shape
img_data = np.squeeze(img_input.dataobj).astype(np.float32)

#%%
Subject = np.array("Test").repeat(np.product(img_shape))
Sex = np.array("Male").repeat(np.product(img_shape))

index = np.indices(img_shape)
Pos_x = index[0].flatten()
Pos_y = index[1].flatten()
Pos_z = index[2].flatten()

Intensity = img_data.flatten()

df = pd.DataFrame({"Subject": Subject,
                   "Sex": Sex,
                   "Pos_X": Pos_x,
                   "Pos_Y": Pos_y,
                   "Pos_Z": Pos_z,
                   "Intensity": Intensity})

#%%
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
)

df_data = df[
    ["Pos_X",
    "Pos_Y",
    "Pos_Z",
    "Intensity"]].values

scaled_df_data = StandardScaler().fit_transform(df_data)

#%%
embedding = reducer.fit_transform(df_data)

#%%

plt.scatter(
    embedding[:, 0],
    embedding[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection', fontsize=24)

#%%

kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(embedding)

# bandwidth = cluster.estimate_bandwidth(df_data, quantile=0.2, n_samples=500)
# ms_labels = cluster.MeanShift(bandwidth=2).fit_predict(df_data)
# knn_graph = kneighbors_graph(df_data, 30, include_self=False)
# agg_labels= cluster.AgglomerativeClustering(linkage="average", connectivity=knn_graph, n_clusters=10).fit_predict(df_data)

#%%
plt.scatter(embedding[:, 0]
            ,embedding[:, 1],
            c = kmeans_labels,
            cmap = "turbo")
plt.gca().set_aspect('equal', 'datalim')
plt.title('kmeans')

#%%

img_labeled = kmeans_labels
img_labeled.shape = img_shape
img_labeled = np.flip(img_labeled,2)

#%%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(ndimage.rotate(img_data[:,40,:],-90), cmap="gray",
               origin='upper')

plt.subplot(1,2,2)
plt.imshow(ndimage.rotate(img_labeled[:,40,:],-90), cmap="jet",
               origin='lower')


plt.show()