import sys
import os
import glob
import math
import gc
import argparse

import numpy as np
from skimage import io
import pandas as pd




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Compute clustering maps and statistics")    

    parser.add_argument("--clustering_assignment_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans")
    parser.add_argument("--n_clusters", type=int, default=16)
    parser.add_argument("--wsi_dim_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv")
    parser.add_argument("--ignore_cluster_ids", type=int, nargs="*", default=None) # [1,8,0]
    parser.add_argument("--root_out_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16")

    args = parser.parse_args()

    clustering_assignment_dir = args.clustering_assignment_dir
    n_clusters = args.n_clusters
    wsi_dim_filepath = args.wsi_dim_filepath
    root_out_dir = args.root_out_dir

    ignore_clusters_arr = None
    if(args.ignore_cluster_ids is not None and len(args.ignore_cluster_ids)>0):
        ignore_clusters_arr = np.array(list(set(args.ignore_cluster_ids)))

    tile_vis_size = 1

    groups_dict = {} # {<group_name>:[<group_indx>, list of cluster ids]}
    for i in range(n_clusters):
        groups_dict[f"c{i}"]=[i,[i]]
    n_groups = n_clusters


    if(not os.path.exists(root_out_dir)):
        os.makedirs(root_out_dir, exist_ok=True)
    

    wsi_dim_df = pd.read_csv(wsi_dim_filepath)
    wsi_dim_arr = wsi_dim_df.to_numpy()

    slide_names = []
    slide_total_patches = []
    slide_n_patches_in_cluster = []
    slide_proportion_patches_in_cluster = []

    maps_out_dir = os.path.join(root_out_dir, "clustering_maps")
    os.makedirs(maps_out_dir, exist_ok=True)

    stats_out_dir = os.path.join(root_out_dir, "clustering_stats")
    os.makedirs(stats_out_dir, exist_ok=True)

    wi = 0
    # for indx in range(wsi_dim_arr.shape[0]):
    for indx in np.random.permutation(wsi_dim_arr.shape[0]):

        slide_name = wsi_dim_arr[indx,0]

        print(indx, slide_name)

        tile_size = -1
        if(wsi_dim_arr.shape[-1] == 3):
            _, width, height = wsi_dim_arr[wsi_dim_arr[:,0]==slide_name][0]
        else:
            _, width, height, mag, tile_size  = wsi_dim_arr[wsi_dim_arr[:,0]==slide_name][0]
        print(f"height={height}")
        print(f"width={width}")

        files = glob.glob(os.path.join(clustering_assignment_dir, f"{slide_name}*.csv"))
        if(files is None or len(files)==0):
            print('No Clustering Info')
            continue
        clustering_filepath = files[0]       
        clustering_df = pd.read_csv(clustering_filepath)

        tiles_coord_x = clustering_df["coord_x"].to_numpy()
        tiles_coord_y = clustering_df["coord_y"].to_numpy()
        tiles_cluster_id = clustering_df["cluster_id"].to_numpy()
        tiles_cluster_dist = clustering_df["cluster_dist"].to_numpy()
        if(tile_size <= 0):
            found_tile_size = False
            ti = 0
            while(ti+1<len(tiles_coord_x)):
                x1 = tiles_coord_x[ti]
                x2 = tiles_coord_x[ti+1]
                y1 = tiles_coord_y[ti]
                y2 = tiles_coord_y[ti+1]
                tile_size = min(abs(x2-x1), abs(y2-y1))   
                if(tile_size > 0 and tile_size < 600):
                    found_tile_size = True
                    break
                tile_size = max(abs(x2-x1), abs(y2-y1))   
                if(tile_size < 600):
                    found_tile_size = True
                    break
                ti += 1
            if(not found_tile_size):
                ti = 0
                while(ti+1<len(tiles_coord_x)):
                    x1 = tiles_coord_x[ti]
                    x2 = tiles_coord_x[ti+1]
                    y1 = tiles_coord_y[ti]
                    y2 = tiles_coord_y[ti+1]
                    tile_size = min(abs(x2-x1), abs(y2-y1))   
                    if(tile_size > 0 and tile_size < 1100):
                        found_tile_size = True
                        break
                    tile_size = max(abs(x2-x1), abs(y2-y1))   
                    if(tile_size < 1100):
                        found_tile_size = True
                        break
                    ti += 1
            if(not found_tile_size):
                ti = 0
                while(ti+1<len(tiles_coord_x)):
                    x1 = tiles_coord_x[ti]
                    x2 = tiles_coord_x[ti+1]
                    y1 = tiles_coord_y[ti]
                    y2 = tiles_coord_y[ti+1]
                    tile_size = min(abs(x2-x1), abs(y2-y1))   
                    if(tile_size > 0 and tile_size < 1600):
                        found_tile_size = True
                        break
                    tile_size = max(abs(x2-x1), abs(y2-y1))   
                    if(tile_size < 1600):
                        found_tile_size = True
                        break
                    ti += 1
            if(not found_tile_size):
                ti = 0
                while(ti+1<len(tiles_coord_x)):
                    x1 = tiles_coord_x[ti]
                    x2 = tiles_coord_x[ti+1]
                    y1 = tiles_coord_y[ti]
                    y2 = tiles_coord_y[ti+1]
                    tile_size = min(abs(x2-x1), abs(y2-y1))   
                    if(tile_size > 0 and tile_size < 2100):
                        found_tile_size = True
                        break
                    tile_size = max(abs(x2-x1), abs(y2-y1))   
                    if(tile_size < 2100):
                        found_tile_size = True
                        break
                    ti += 1
        scale = tile_vis_size/tile_size
        new_height = int(height*scale+1)
        new_width = int(width*scale+1)

        if(ignore_clusters_arr is not None and len(ignore_clusters_arr)>0):
            clustering_df = clustering_df[~clustering_df['cluster_id'].isin(ignore_clusters_arr)] 

            tiles_coord_x = clustering_df["coord_x"].to_numpy()
            tiles_coord_y = clustering_df["coord_y"].to_numpy()
            tiles_cluster_id = clustering_df["cluster_id"].to_numpy()
            tiles_cluster_dist = clustering_df["cluster_dist"].to_numpy()

        tiles_coord_x_scaled = np.round(tiles_coord_x*scale).astype(int)
        tiles_coord_y_scaled = np.round(tiles_coord_y*scale).astype(int)
        tiles_coord_x_scaled[tiles_coord_x_scaled>=new_width ] = new_width-1
        tiles_coord_y_scaled[tiles_coord_y_scaled>=new_height ] = new_height-1

        cluster_id_map = np.ones((new_height, new_width))*-1
        for ti in range(len(tiles_coord_x_scaled)):
            cluster_id_map[tiles_coord_y_scaled[ti]:min(new_height, tiles_coord_y_scaled[ti]+tile_vis_size), tiles_coord_x_scaled[ti]:min(new_width, tiles_coord_x_scaled[ti]+tile_vis_size)] = tiles_cluster_id[ti]
        foreground_mask = cluster_id_map>=0
        background_mask = cluster_id_map<0        
        io.imsave( os.path.join(maps_out_dir, f"{slide_name}_fg_mask.png"), (foreground_mask*255).astype(np.uint8), check_contrast=False)
        io.imsave( os.path.join(maps_out_dir, f"{slide_name}_bg_mask.png"), (background_mask*255).astype(np.uint8), check_contrast=False)
        cluster_id_map.astype(np.uint8).dump(os.path.join(maps_out_dir, f"{slide_name}_cluster_id_map.npy"))


        total_patches = tiles_coord_x.shape[0]
        n_patches_in_cluster = np.zeros((1,n_clusters))
        proportion_patches_in_cluster = np.zeros((1,n_clusters))
        for ci in range(n_clusters):
            n_patches_in_cluster[0,ci] = (tiles_cluster_id==ci).sum()
            proportion_patches_in_cluster[0,ci] = n_patches_in_cluster[0,ci]/total_patches
        slide_names.append(slide_name)
        slide_total_patches.append(total_patches)
        slide_n_patches_in_cluster.append(n_patches_in_cluster)
        slide_proportion_patches_in_cluster.append(proportion_patches_in_cluster)

        wi += 1
        sys.stdout.flush()

    out_filepath = os.path.join(stats_out_dir, f"cluster_stats.csv")
    slide_n_patches_in_cluster = np.concatenate(slide_n_patches_in_cluster, axis=0)
    slide_proportion_patches_in_cluster = np.concatenate(slide_proportion_patches_in_cluster, axis=0)
    cluster_ids_ordered = np.argsort(slide_n_patches_in_cluster, axis=1)[:,::-1]
    df = pd.DataFrame()
    df['slide_name'] = slide_names
    df['total_patches'] = slide_total_patches
    for ci in range(n_clusters):
        df[f'count_{ci}'] = slide_n_patches_in_cluster[:,ci]
    for ci in range(n_clusters):
        df[f'proportion_{ci}'] = slide_proportion_patches_in_cluster[:,ci]
    for ci in range(n_clusters):
        df[f'rank_{ci}'] = cluster_ids_ordered[:,ci]
    df.to_csv(out_filepath,index=False)
        