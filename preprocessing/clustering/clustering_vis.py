import sys
import os
import glob
import math
import gc
import argparse

import numpy as np
from skimage import io
import pandas as pd


def hex_to_rgb(hex_color_list):
    rgb_color_array = np.zeros((len(hex_color_list), 3))
    for indx, hex_color_code in enumerate(hex_color_list):
        hex_color_code = hex_color_code.lstrip('#')  # Remove '#' if present

        if len(hex_color_code) != 6:
            return None  # Invalid hex code length

        try:
            r = int(hex_color_code[0:2], 16)
            g = int(hex_color_code[2:4], 16)
            b = int(hex_color_code[4:6], 16)
            rgb_color_array[indx] = [r, g, b]
        except ValueError:
            return None  # Invalid hexadecimal characters
    return rgb_color_array

def load_csv_as_df(filepath, columns=None):
    if(columns is None):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, header=0, usecols=columns)
    return df

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Compute foreground mask from patch clusters")    

    parser.add_argument("--clustering_assignment_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans")
    parser.add_argument("--n_clusters", type=int, default=16)
    parser.add_argument("--wsi_dim_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv")
    parser.add_argument("--ignore_cluster_ids", type=int, nargs="+", default=None) # [1,8,0]
    parser.add_argument("--root_out_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans_vis")

    args = parser.parse_args()

    clustering_assignment_dir = args.clustering_assignment_dir
    n_clusters = args.n_clusters
    wsi_dim_filepath = args.wsi_dim_filepath
    root_out_dir = args.root_out_dir

    ignore_clusters_arr = None
    if(args.ignore_cluster_ids is not None and len(args.ignore_cluster_ids)>0):
        ignore_clusters_arr = np.array(list(set(args.ignore_cluster_ids)))

    tile_vis_size = 50

    # https://lospec.com/palette-list/basic-16
    colors16 = [
        '#f2c0a2',
        '#e98472',
        '#d82323',
        '#98183c',
        '#1fcb23',
        '#126d30',
        '#26dddd',
        '#1867a0',
        '#934226',
        '#6c251e',
        '#f7e26c',
        '#edb329',
        '#e76d14',
        '#f2f2f9',
        '#6a5fa0',
        '#161423',
        ]

    # https://lospec.com/palette-list/category-colors
    colors24 = [
        '#75adc0',
        '#29758e',
        '#12617b',
        '#dcb7c0',
        '#ba8591',
        '#9d5c6b',
        '#c0f1f1',
        '#54bcbd',
        '#149090',
        '#ffe197',
        '#e3b23b',
        '#be8245',
        '#bbd1ae',
        '#91b17f',
        '#6f9657',
        '#e88c7d',
        '#be5645',
        '#9e3423',
        '#b8c3e2',
        '#8091c4',
        '#546ba6',
        '#cba481',
        '#ab7c64',
        '#85563e',
    ]

    groups_dict = {} # {<group_name>:[<group_indx>, list of cluster ids]}
    for i in range(n_clusters):
        groups_dict[f"c{i}"]=[i,[i]]
    n_groups = n_clusters

    if(not os.path.exists(root_out_dir)):
        os.makedirs(root_out_dir, exist_ok=True)

    if(n_clusters <=16):
        cluster_color_arr = hex_to_rgb(colors16)
    else:
        cluster_color_arr = hex_to_rgb(colors24)
    

    wsi_dim_df = pd.read_csv(wsi_dim_filepath)
    wsi_dim_arr = wsi_dim_df.to_numpy()

    slide_names = []
    slide_total_patches = []
    slide_n_patches_in_cluster = []
    slide_proportion_patches_in_cluster = []

    wi = 0
    # for indx in range(wsi_dim_arr.shape[0]):
    for indx in np.random.permutation(wsi_dim_arr.shape[0]):

        slide_name = wsi_dim_arr[indx,0]

        print(indx, slide_name)
        vis_out_filepath = os.path.join(root_out_dir, f"{slide_name}_vis_clusters.png")
        if(os.path.exists(vis_out_filepath)):
            continue

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

        vis_map = np.zeros((new_height, new_width, 3))

        if(ignore_clusters_arr is not None):
            clustering_df = clustering_df[~clustering_df['cluster_id'].isin(ignore_clusters_arr)] 

            tiles_coord_x = clustering_df["coord_x"].to_numpy()
            tiles_coord_y = clustering_df["coord_y"].to_numpy()
            tiles_cluster_id = clustering_df["cluster_id"].to_numpy()
            tiles_cluster_dist = clustering_df["cluster_dist"].to_numpy()

        tiles_coord_x_scaled = np.round(tiles_coord_x*scale).astype(int)
        tiles_coord_y_scaled = np.round(tiles_coord_y*scale).astype(int)
        tiles_coord_x_scaled[tiles_coord_x_scaled>=new_width ] = new_width-1
        tiles_coord_y_scaled[tiles_coord_y_scaled>=new_height ] = new_height-1

        for ti in range(len(tiles_coord_x_scaled)):
            vis_map[tiles_coord_y_scaled[ti]:min(new_height, tiles_coord_y_scaled[ti]+tile_vis_size), tiles_coord_x_scaled[ti]:min(new_width, tiles_coord_x_scaled[ti]+tile_vis_size)] = cluster_color_arr[tiles_cluster_id[ti]]
        io.imsave(vis_out_filepath, vis_map.astype(np.uint8), check_contrast=False)


        wi += 1
        sys.stdout.flush()

