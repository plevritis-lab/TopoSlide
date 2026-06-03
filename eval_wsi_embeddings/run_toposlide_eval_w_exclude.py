import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import pandas as pd
import numpy as np
import h5py
import glob
import os
import traceback
import pickle
import argparse
# import skimage.io as io

import torch
from model.build_toposlide import TopoSlide
from model.configuration import TopoSlideConfig

# lock_filename = "processing.txt"
# done_filename = "done.txt"
# error_filename = "error.txt"

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Eval TopoSlide")    

    parser.add_argument("--patch_embedding_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/patches_20x_512_conch_tile_embedding")
    parser.add_argument("--slide_root_output_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/tcga_luad_toposlide_wsi_20x_512")
    parser.add_argument("--patch_size_lv0", type=int, default=512)
    parser.add_argument("--checkpoint_path", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/trained_models/toposlide_tcga_luad.pt")
    parser.add_argument("--wsi_dim_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/tcga_luad_wsi_dim.csv")
    parser.add_argument("--clustering_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans")
    parser.add_argument("--ignore_cluster_ids", type=int, nargs="*", default=[1,8,0,15])
    parser.add_argument("--n_clusters", type=int, default=16)

    args = parser.parse_args()


    patch_embedding_dir = args.patch_embedding_dir
    slide_root_output_dir = args.slide_root_output_dir
    patch_size_lv0 = args.patch_size_lv0
    checkpoint_path = args.checkpoint_path
    clustering_dir = args.clustering_dir
    wsi_dim_filepath = args.wsi_dim_filepath
    n_clusters = args.n_clusters
    ignore_clusters_arr = np.array(list(set(args.ignore_cluster_ids)))

    if(patch_size_lv0<=0 ):
        raise Exception(f"patch_size_lv0={patch_size_lv0} must be greater than zero")
    if(n_clusters<=0 ):
        raise Exception(f"n_clusters={n_clusters} must be greater than zero")
    if(ignore_clusters_arr is not None and len(ignore_clusters_arr)>0 and ( min(ignore_clusters_arr)<0 or max(ignore_clusters_arr)>=n_clusters)):
        raise Exception(f"ignore_clusters_arr={ignore_clusters_arr} not in range [0,n_clusters={n_clusters}]")

    tile_vis_size = 1

    patch_clustering_files = np.array(glob.glob(os.path.join(clustering_dir, f"*.csv")) )
    patch_clustering_slide_names = np.array([os.path.basename(filepath).split('_')[0] for filepath in patch_clustering_files])

    os.makedirs(slide_root_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)

    config = TopoSlideConfig()
    config.vision_config.pos_encode_type = 'abs'
    print('config done')
    model = TopoSlide(config)
    print('model architecture created')

    # state_dict = {}
    # with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
    #     # Iterate through the keys (tensor names) in the safetensors file
    #     for key in f.keys():
    #         # Get each tensor by its key and store it in the state_dict
    #         state_dict[key] = f.get_tensor(key)

    state_dict2 = torch.load(checkpoint_path)
    state_dict = state_dict2['toposlide'] 

    # for key in state_dict.keys():
    #     print(key)
    print('stat_dict loaded')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    print('model loaded')
    print('model.device', model.device)


    wsi_dim_df = pd.read_csv(wsi_dim_filepath)
    wsi_dim_arr = wsi_dim_df.to_numpy()

    for indx in np.random.permutation(wsi_dim_arr.shape[0]):
        slide_name = wsi_dim_arr[indx,0]
        print('slide_name', slide_name)
        _, width, height, mag, pw  = wsi_dim_arr[wsi_dim_arr[:,0]==slide_name][0]
        patch_size_lv0 = pw
        patch_feat_filepath = glob.glob(os.path.join(patch_embedding_dir, f"{slide_name}*.h5"))
        print('patch_feat_filepath', patch_feat_filepath)
        if(patch_feat_filepath is None or len(patch_feat_filepath )==0):
            continue
        patch_feat_filepath = patch_feat_filepath[0]
        patch_clustering_filepath = glob.glob(os.path.join(clustering_dir, f"{slide_name}*.csv"))
        print('patch_clustering_filepath', patch_clustering_filepath)
        if(patch_clustering_filepath is None or len(patch_clustering_filepath )==0):
            continue
        patch_clustering_filepath = patch_clustering_filepath[0]
        slide_filepath = os.path.join(slide_root_output_dir, os.path.basename(patch_feat_filepath)[:-len(".h5")]+".pkl")
        print('slide_filepath', slide_filepath)
        if(os.path.exists(slide_filepath)):
            continue
        print('slide_name', slide_name)
        patch_embedding_hdf_file = h5py.File(patch_feat_filepath, 'r')
        embedding_arr = patch_embedding_hdf_file['features'][:]
        coord_arr = patch_embedding_hdf_file['coords'][:]

        clustering_df = pd.read_csv(patch_clustering_filepath)
        if(ignore_clusters_arr is not None and len(ignore_clusters_arr)>0):
            clustering_df = clustering_df[~clustering_df['cluster_id'].isin(ignore_clusters_arr)] 
        tiles_coord_x = clustering_df["coord_x"].to_numpy()
        tiles_coord_y = clustering_df["coord_y"].to_numpy()
        tiles_cluster_id = clustering_df["cluster_id"].to_numpy()
        # tiles_cluster_dist = clustering_df["cluster_dist"].to_numpy()
        print('tiles_coord_x', tiles_coord_x.shape)
        print('coord_arr', coord_arr.shape)
        if(tiles_coord_x.shape[0]==0):
            continue

        # ti = 0
        # while(True):
        #     x1 = tiles_coord_x[ti]
        #     x2 = tiles_coord_x[ti+1]
        #     y1 = tiles_coord_y[ti]
        #     y2 = tiles_coord_y[ti+1]
        #     tile_size = max(x2-x1, y2-y1)   
        #     if(tile_size < 1500):
        #         break
        #     ti += 1
        tile_size = pw
        scale = tile_vis_size/tile_size

        new_height = int(height*scale+1)
        new_width = int(width*scale+1)


        coord_arr_scaled = np.round(coord_arr*scale).astype(int)
        coord_arr_scaled_x = coord_arr_scaled[:, 0]
        coord_arr_scaled_y = coord_arr_scaled[:, 1]
        coord_arr_scaled_x[coord_arr_scaled_x >= new_width] = new_width-1
        coord_arr_scaled_y[coord_arr_scaled_y >= new_height] = new_height-1

        tiles_coord_scaled_x = np.round(tiles_coord_x*scale).astype(int)
        tiles_coord_scaled_y = np.round(tiles_coord_y*scale).astype(int)
        tiles_coord_scaled_x[tiles_coord_scaled_x >= new_width] = new_width-1
        tiles_coord_scaled_y[tiles_coord_scaled_y >= new_height] = new_height-1

        cluster_id_map = np.ones((new_height, new_width))*-1
        for ti in range(len(tiles_coord_scaled_x)):
            if(ignore_clusters_arr is not None and tiles_cluster_id[ti] in ignore_clusters_arr):
                continue
            cluster_id_map[tiles_coord_scaled_y[ti]:min(new_height, tiles_coord_scaled_y[ti]+tile_vis_size), tiles_coord_scaled_x[ti]:min(new_width, tiles_coord_scaled_x[ti]+tile_vis_size)] = tiles_cluster_id[ti]

        embedding_im = np.zeros((new_height, new_width, embedding_arr.shape[-1] ))
        embedding_im[(coord_arr_scaled_y, coord_arr_scaled_x)] = embedding_arr

        coord_map = np.zeros((new_height, new_width, 2 ))
        coord_map[(coord_arr_scaled_y, coord_arr_scaled_x,0)] = coord_arr[:,0]
        coord_map[(coord_arr_scaled_y, coord_arr_scaled_x,1)] = coord_arr[:,1]


        features = embedding_im[cluster_id_map>=0][np.newaxis, :].astype(np.float16)
        coords = coord_map[cluster_id_map>=0][np.newaxis, :].astype(int)
        print('features', features.shape, features.dtype)
        print('coords', coords.shape, coords.dtype)



        # extract slide embedding
        slide_filepath = os.path.join(slide_root_output_dir, os.path.basename(patch_feat_filepath)[:-len(".h5")]+".pkl")
        if(os.path.exists(slide_filepath)):
            continue
        with torch.autocast('cuda', torch.float16), torch.inference_mode():
            if(len(features.shape)!=3):
                features = features[np.newaxis, :]
            if(len(coords.shape)!=3):
                coords = coords[np.newaxis, :]
            features = torch.from_numpy(features).to(device)
            coords = torch.from_numpy(coords).to(device)
            slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
            with open(slide_filepath, 'wb') as file:
                pickle.dump(slide_embedding, file)
        # break