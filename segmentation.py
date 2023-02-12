import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        clusters = [[] for _ in range(k)]     # To help calculate the new centers
        for i in range(N):
            distance = ["lol" for _ in range (k)]  # To store distances for each point with each center
            for j in range(k):
                distance[j] = np.sqrt(sum((centers[j] - features[i])**2))  # Calcualte distance

            min_soFar = np.Infinity
            min_index = "lolz"
            for l in range(k):
                if(distance[l]<min_soFar):
                    min_soFar = distance[l]
                    min_index = l
            clusters[min_index].append(features[i].tolist()) # Add the row to the corresponding closest center
            assignments[i] = min_index                     # label the closest center to the current row

        new_centers = ["lol" for _ in range (k)]

        for index,cluster in enumerate(clusters):
            cluster_sum = [0 for _ in range(D)]
            cluster_avg = [0 for _ in range(D)]
            for point in cluster:
                for m in range(D):
                    cluster_sum[m] = cluster_sum[m] + point[m]     # sum of points in each cluster
            length = 0

            for point in cluster:
                length = length + 1                  # number of points in each cluster xD
            for f_Index,feature in enumerate(cluster_avg):
                cluster_avg[f_Index] = cluster_sum[f_Index] / length
            new_centers[index] = cluster_avg
        flag = True

        for sad in range(k):
            for sadder in range(D):
                temp1 = new_centers[sad]
                temp2 = centers[sad]
                if(not(temp1[sadder] == temp2[sadder])):
                    flag = False
        if (flag):
            return assignments
        centers = new_centers


    return assignments

def kmeans_fast(input,k=3,num_iteration=100):

    N, D = input.shape
    assert N >= k

    iter = 0
    idx = np.random.choice(len(input), size=k, replace=False)
    centroids_old = np.array([input[x,:] for x in idx])
    membership = np.zeros(len(input),dtype=np.int8)
    while(iter<num_iteration):
        points = [[] for _ in range(len(idx))]
        distances = np.array([np.linalg.norm(input-centroid,axis=1) for centroid in centroids_old])
        res = np.argmin(np.transpose(distances),axis=1)
        for i,label in enumerate(res):
            points[label].append(input[i,:])
            membership[i]=label
        centroids_new =np.array([np.mean(x,axis=0) for x in points])
        iter += 1
        if (np.array_equal(centroids_new,centroids_old)):
            break
        centroids_old = centroids_new
    #print("Number of Iterations is  ",iter)

    return membership





### Pixel-Level Features
def color_features(img):
    H, W, C = img.shape
    img = img_as_float(img)

    return img.reshape(H*W,C)



def color_position_features(img):

    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))


    temp_matrix = np.zeros((H,W))
    indices = np.triu_indices_from(temp_matrix,k=-max(H,W))
    indices_res = np.transpose(np.array(indices).reshape(2,-1))
    features[:, 0 : C] = color.reshape((H * W, C))
    features[:, C : C + 2] = indices_res
    features = (features - np.mean(features, axis = 0)) / np.std(features, axis = 0)

    return features

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):

    return np.mean(mask==mask_gt)



def evaluate_segmentation(mask_gt, segments):

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
