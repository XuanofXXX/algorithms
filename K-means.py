#%%
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import pickle as pkl
from sklearn.decomposition import PCA
# %%

def generate_vectors():
    animals = ['Cats', 'Dogs']
    for a in animals:
        folder_path = f'./archive/DogsCats/{a}/'
        vectors = []
        # loop through all files in the folder
        for filename in os.listdir(folder_path):
            # get the full file path
            file_path = os.path.join(folder_path, filename)
            # check if the file is a regular file (not a folder)
            if os.path.isfile(file_path):
                # do something with the file
                files = {"image": open(file_path, "rb")}
                url = "http://bl.mmd.ac.cn:8889/image_query"
                
                r = requests.post(url, files=files)
                resp = r.json()
                vectors.append(resp['embedding'])
    return vectors

# %%
# calculate the distance between two points
def distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
#%%
# K-means algorithm¡
def k_means(points, k, max_iterations=100):
    # randomly initialize k centroids
    centroids = points[np.random.choice(points.shape[0], k, replace=False), :]

    # initialize variables
    iterations = 0
    old_centroids = np.zeros(centroids.shape)

    # run K-means algorithm
    while iterations < max_iterations and not np.array_equal(centroids, old_centroids):
        # assign each point to the nearest centroid
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmax(distances)
            clusters[cluster_index].append(point)

        # update centroids
        old_centroids = centroids.copy()
        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(cluster, axis=0)

        iterations += 1

    return clusters, centroids
#%%
# 降维
def reduce_dim(vectors):
    pca = PCA(n_components=2)
    newVec = pca.fit_transform(vectors)
    return newVec
#%% 
vectors = []
vectors_cat = vectors_dog = []
with open("./archive/my_cat_list.pkl", 'rb') as f:
    vectors_cat = pkl.load(f)
with open("./archive/my_dog_list.pkl", 'rb') as f:
    vectors_dog = pkl.load(f)
vectors_cat = np.array(vectors_cat)
vectors_dog = np.array(vectors_dog)
vectors = np.vstack((vectors_cat, vectors_dog))
vectors = np.array(vectors)

vectors_cat, vectors_dog = reduce_dim(vectors)[0:54, :], reduce_dim(vectors)[54:, :]

#%%
if __name__ == "__main__":
    # read points
    # run K-means algorithm
    clusters, centroids = k_means(vectors, 2, 1000)

    centroids = np.array(centroids)
    c0 = clusters[0]
    len0 = len(c0)
    c1 = clusters[1]
    cc = np.vstack((centroids, np.array(c0),np.array(c1),))
    cc = reduce_dim(cc)

    centroids = cc[:2, :];
    clusters[0] = cc[2:2+len0, :];
    clusters[1] = cc[2+len0:, :];
    
    # plot results
    colors = ['r', 'g']
    for i, cluster in enumerate(clusters):
        
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i])
    centroids = reduce_dim(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    plt.show()
    
# %%
cluster0 = clusters[0]
cluster1 = clusters[1]

cosine_sims1 = [distance(vectors_cat[i], cluster0[i]) for i in range(54)]
cosine_sims2 = [distance(vectors_cat[i], cluster1[i]) for i in range(54)]
cosine_sims3 = [distance(vectors_dog[i], cluster0[i]) for i in range(54)]
cosine_sims4 = [distance(vectors_dog[i], cluster1[i]) for i in range(54)]
print("Cosine Similarities: clu1 in cats", sum(cosine_sims1) / 54)
print("Cosine Similarities: clu2 in cats", sum(cosine_sims2) / 54)
print("Cosine Similarities: clu1 in dogs", sum(cosine_sims3) / 54)
print("Cosine Similarities: clu2 in dogs", sum(cosine_sims4) / 54)

# %%
with open("./archive/vectors.pkl", 'wb') as f:
    pkl.dump(vectors, f)
    
# %%
vectors_cat = vectors_dog = []
with open("./archive/my_cat_list.pkl", 'rb') as f:
    vectors_cat = pkl.load(f)
with open("./archive/my_dog_list.pkl", 'rb') as f:
    vectors_dog = pkl.load(f)
vectors_cat = np.array(vectors_cat)
vectors_dog = np.array(vectors_dog)
vectors = vectors_cat + vectors_dog
vectors = np.array(vectors)

# %%
vectors_cat = reduce_dim(vectors_cat)
vectors_dog = reduce_dim(vectors_dog)

# %%
# rename all files in a folder
# from .jpeg to .jpg
folder_path = './archive/DogsCats/Dogs/'
for filename in os.listdir(folder_path):
    if filename.endswith('.jpeg'):
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, filename.replace('.jpeg', '.jpg'))
        os.rename(old_path, new_path)
# %%
