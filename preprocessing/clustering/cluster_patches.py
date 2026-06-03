import sys
import os
import glob
import traceback 
import math
import argparse

import numpy as np
import pandas as pd
import h5py
from sklearn.neighbors import NearestNeighbors

from clustering_utils import cluster_affinity_propagation, cluster_kmeans



def load_features(root_tile_embeddings):
    feat_hdf_files = glob.glob(os.path.join(root_tile_embeddings, "*.h5"))
    print('feat_hdf_files', len(feat_hdf_files))

    features_list = []
    for indx, feat_hdf_filepath in enumerate(feat_hdf_files):
        print(indx, feat_hdf_filepath, end='\r')
        hdf_file = h5py.File(feat_hdf_filepath, 'r')
        embedding_arr = hdf_file['features'][:]
        coord_arr = hdf_file['coords'][:]
        features_list.append(embedding_arr)

    features_arr = np.concatenate(features_list, axis=0)
    print('features_arr', features_arr.shape)
    return features_arr


def predict_cluster(root_tile_embeddings, cluster_centers, cluster_centers_filepath, root_out_dir, suffix=""):
    if(cluster_centers is None):
        cluster_centers = np.load(cluster_centers_filepath, allow_pickle=True)
    n_clusters = cluster_centers.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_clusters, algorithm='brute', metric='cosine').fit(cluster_centers)

    feat_hdf_files = glob.glob(os.path.join(root_tile_embeddings, "*.h5"))
    print('feat_hdf_files', len(feat_hdf_files))

    out_dir = os.path.join(root_out_dir, f"prediction_{clustering_method_name}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
        

    for feat_hdf_filepath in feat_hdf_files:
        hdf_file = h5py.File(feat_hdf_filepath, 'r')
        embedding_arr = hdf_file['features'][:]
        coord_arr = hdf_file['coords'][:]

        out_filepath = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(feat_hdf_filepath))[0]}.csv")
        if(os.path.exists(out_filepath)):
            continue
        distances, indices = nbrs.kneighbors(embedding_arr)
        print('distances', distances.shape)
        print('indices', indices.shape)
        sort_indices = np.argsort(indices, axis=1)
        distances_ordered_by_cluster_id = np.take_along_axis(distances, sort_indices, axis=1) 
        print('distances_ordered_by_cluster_id', distances_ordered_by_cluster_id.shape)
        distances_ordered_by_cluster_id = distances_ordered_by_cluster_id.reshape(distances.shape)
        print('distances_ordered_by_cluster_id', distances_ordered_by_cluster_id.shape)
        df = pd.DataFrame()
        df['coord_x'] = coord_arr[:,0]
        df['coord_y'] = coord_arr[:,1]
        df['cluster_id'] = indices[:,0]
        df['cluster_dist'] = distances[:,0]
        for i in range(n_clusters):
            df[f'dist_{i}'] = distances_ordered_by_cluster_id[:,i]
        df.to_csv(out_filepath,index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Cluster patch embeddings")    

    parser.add_argument("--root_tile_embeddings_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding")
    parser.add_argument("--n_clusters_preference", type=int, default=16)
    parser.add_argument("--clustering_method_name", type=str, default="kmeans", help="supported options: kmeans, afp")
    parser.add_argument("--root_out_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16")
    parser.add_argument("--cluster_center_filepath", type=str, default=None, help="if not None, will be used to predict clustering")
    
    args = parser.parse_args()

    root_tile_embeddings_dir = args.root_tile_embeddings_dir
    n_clusters_preference = args.n_clusters_preference
    clustering_method_name = args.clustering_method_name
    root_out_dir = args.root_out_dir
    cluster_center_filepath = args.cluster_center_filepath


    if(not os.path.exists(root_out_dir )):
        os.makedirs(root_out_dir, exist_ok=True)

    if(cluster_center_filepath is None):
        
        features_arr = load_features(root_tile_embeddings_dir)
        print('features_arr', features_arr.shape)

        if(clustering_method_name == "afp"):
            _, cluster_centers = cluster_affinity_propagation(features_arr, None, root_out_dir, f"", vis = None, perplexity=5, save_distances=False)
        elif(clustering_method_name == "kmeans"):
            _, cluster_centers = cluster_kmeans(features_arr, n_clusters_preference, None, root_out_dir, f"", pca_init=True, vis = None, perplexity=5, save_distances=False)
        else:
            print("clustering_method_name", clustering_method_name, "not supported")
            exit()
        print('finished clustering')
        print('cluster_centers', cluster_centers.shape)
        predict_cluster(root_tile_embeddings_dir, cluster_centers, None, root_out_dir, suffix=f"")
        print('finished prediction')
    else:
        predict_cluster(root_tile_embeddings_dir, None, cluster_center_filepath, root_out_dir, suffix=f"")
        print('finished prediction')
