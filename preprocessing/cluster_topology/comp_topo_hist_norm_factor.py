import sys
import os
import glob
import traceback
import argparse
import math

import numpy as np
import h5py
import pandas as pd
import skimage.io as io
import cv2


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Compute topology persistence statistics")    

    parser.add_argument("--root_input_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo")
    parser.add_argument("--out_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_hist")
    parser.add_argument("--num_clusters", type=int, default=16)
    parser.add_argument("--wsi_dim_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv")

    args = parser.parse_args()

    root_dir = args.root_input_dir
    out_dir = args.out_dir
    n_clusters = args.num_clusters
    wsi_dim_filepath = args.wsi_dim_filepath

    groups_dict = {} # {<group_name>:[<group_indx>, list of cluster ids]}
    for i in range(n_clusters):
        groups_dict[f"c{i}"]=[i,[i]]

    # groups_dict[f"alveoli"]=[0,[0]]
    # groups_dict[f"c1"]=[1,[1]]
    # groups_dict['lym'] = [2, [6]]

    # groups_dict[f"stroma_1"]=[3,[2]]
    # groups_dict[f"stroma_2"]=[4,[5]]
    # groups_dict[f"stroma_3"]=[5,[7]]
    # groups_dict[f"stroma_4"]=[6,[10]]
    # groups_dict[f"stroma_5"]=[7,[12]]

    # groups_dict[f"tumor_acinar_1"]=[8,[3]]
    # groups_dict[f"tumor_acinar_2"]=[9,[9]]

    # groups_dict[f"tumor_solid_1"]=[10,[4]]
    # groups_dict[f"tumor_solid_2"]=[11,[11]]

    # groups_dict[f"tumor_micropapillary"] = [12, [13]]
    # groups_dict[f"tumor_other"] = [13, [15]]

    # # groups_dict['stroma_all'] = [14, [groups_dict[f"stroma_1"][1][0],
    # #                                     groups_dict[f"stroma_2"][1][0],
    # #                                     groups_dict[f"stroma_3"][1][0],
    # #                                     groups_dict[f"stroma_4"][1][0],
    # #                                     groups_dict[f"stroma_5"][1][0],
    # #                                     ]]
    # # groups_dict['tumor_all'] = [15, [groups_dict[f"tumor_acinar_1"][1][0],
    # #                                     groups_dict[f"tumor_acinar_2"][1][0],
    # #                                     groups_dict[f"tumor_solid_1"][1][0],
    # #                                     groups_dict[f"tumor_solid_2"][1][0],
    # #                                     groups_dict[f"tumor_micropapillary"][1][0],
    # #                                     groups_dict[f"tumor_other"][1][0],
    # #                                     ]]
    # # groups_dict['acinar_all'] = [16, [groups_dict[f"tumor_acinar_1"][1][0],
    # #                                     groups_dict[f"tumor_acinar_2"][1][0],
    # #                                     ]]
    # # groups_dict['solid_all'] = [17, [groups_dict[f"tumor_solid_1"][1][0],
    # #                                     groups_dict[f"tumor_solid_2"][1][0],
    # #                                     ]]
    # groups_dict[f"c14"] = [18, [14]]
    
        
    hist_buckets_arr = np.array([2, 5,10,20,30,40,50, math.inf])
    tile_vis_size = 1
    pers_hist_cc_max = np.zeros((len(hist_buckets_arr)-1))
    pers_hist_holes_max = np.zeros((len(hist_buckets_arr)-1))

    if(not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)
    out_filepath = os.path.join(out_dir, f"topo_pers_hist_stats.txt")

    # slides = os.listdir(root_dir)

    wsi_dim_df = pd.read_csv(wsi_dim_filepath)
    wsi_dim_arr = wsi_dim_df.to_numpy()

    wi = 0
    # for indx in range(wsi_dim_arr.shape[0]):
    for indx in np.random.permutation(wsi_dim_arr.shape[0]):
        slide_name = wsi_dim_arr[indx,0]
    
        files = glob.glob(os.path.join(root_dir, f"{slide_name}_topo.hdf5"))
        if(files is None or len(files)==0):
            print(slide_name, 'No Topo file found')
            continue
        topo_hd_filepath = files[0]       
        topo_hdf_file = h5py.File(topo_hd_filepath, 'r')
        topo_meta_filepath = topo_hd_filepath.replace(".hdf5", "_meta.csv")
        topo_meta_df = pd.read_csv(topo_meta_filepath)
        topo_filenames_list = topo_meta_df["key_name"].to_numpy()

        width = wsi_dim_arr[indx,1]
        height = wsi_dim_arr[indx,2]
        mag = wsi_dim_arr[indx,3]
        pw = wsi_dim_arr[indx,4]
        # print(wi, slide_name, 'width, height', width, height)
        tile_size = pw
        scale = tile_vis_size/tile_size
        height_scaled = height * scale
        width_scaled = width * scale

        for group_name, val in groups_dict.items():
            gi = val[0]
            cluster_list = val[1]
            
            pnt_cloud_name = f"pnts_cc_{group_name}_g{gi}"
            filename_pattern = f"{pnt_cloud_name}_pnts_cc_exclude_bg_topo_cps_original_coord.csv"
            r=pd.Series(topo_filenames_list).str.match(filename_pattern)
            selected_keys = topo_filenames_list[r]
            if(len(selected_keys)>0):
                print('gi', gi)
                csv_ds = topo_hdf_file[filename_pattern]
                pd_cc_df = pd.DataFrame(csv_ds['data'][:], index=csv_ds['index'][:], columns=csv_ds['columns'][:].astype(str))
                cc_bcp_y = (pd_cc_df['bcp_y'].to_numpy()*scale).clip(0,height_scaled-1)
                cc_bcp_x = (pd_cc_df['bcp_x'].to_numpy()*scale).clip(0,width_scaled-1)
                cc_dcp_y = (pd_cc_df['dcp_y'].to_numpy()*scale).clip(0,height_scaled-1)
                cc_dcp_x = (pd_cc_df['dcp_x'].to_numpy()*scale).clip(0,width_scaled-1)
                # cc_bcp_val = pd_cc_df['bcp_val'].to_numpy() 
                # cc_dcp_val = pd_cc_df['dcp_val'].to_numpy() 
                # cc_pers_val = pd_cc_df['pers'].to_numpy() 
                # Because we use dtm, use euclidean distance between bcp coord and dcp coord as the persistence
                # cc_pers_val = np.linalg.norm(pd_cc_df[['bcp_y','bcp_x']].to_numpy() - pd_cc_df[['dcp_y','dcp_x']].to_numpy(), axis=1)
                cc_pers_val=np.sqrt(((np.stack([cc_bcp_y, cc_bcp_x], axis=-1) - np.stack([cc_dcp_y, cc_dcp_x], axis=-1))**2).sum(axis=1))
                # print('cc_pers_val', cc_pers_val.shape)
                # print('cc_bcp_y', cc_bcp_y.shape)
                # print('np.stack([cc_bcp_y, cc_bcp_x], axis=-1)', np.stack([cc_bcp_y, cc_bcp_x], axis=-1).shape)
                # print('np.stack([cc_dcp_y, cc_dcp_x], axis=-1)', np.stack([cc_dcp_y, cc_dcp_x], axis=-1).shape)

                pers_cc_df = pd.DataFrame(cc_pers_val)
                pers_hist_cc_df = pd.cut(pers_cc_df[0], hist_buckets_arr, right=True).value_counts(sort=False)
                pers_hist_cc_max = np.maximum(pers_hist_cc_max, pers_hist_cc_df.values)
                print('cc_pers_val', cc_pers_val)
                print('pers_hist_cc_df.values', pers_hist_cc_df.values)


            pnt_cloud_name = f"pnts_holes_{group_name}_g{gi}"
            filename_pattern = f"{pnt_cloud_name}_pnts_holes_exclude_bg_filter_hull_topo_cps_original_coord.csv"
            r=pd.Series(topo_filenames_list).str.match(filename_pattern)
            selected_keys = topo_filenames_list[r]
            if(len(selected_keys)>0):
                csv_ds = topo_hdf_file[f"{pnt_cloud_name}_pnts_holes_exclude_bg_filter_hull_topo_cps_original_coord.csv"]
                # csv_ds = topo_hdf_file[f"{pnt_cloud_name}_pnts_holes_exclude_bg_topo_cps.csv"]
                pd_holes_df = pd.DataFrame(csv_ds['data'][:], index=csv_ds['index'][:], columns=csv_ds['columns'][:].astype(str))
                holes_bcp_y = (pd_holes_df['bcp_y'].to_numpy()*scale).clip(0,height_scaled-1)
                holes_bcp_x = (pd_holes_df['bcp_x'].to_numpy()*scale).clip(0,width_scaled-1)
                holes_dcp_y = (pd_holes_df['dcp_y'].to_numpy()*scale).clip(0,height_scaled-1)
                holes_dcp_x = (pd_holes_df['dcp_x'].to_numpy()*scale).clip(0,width_scaled-1) 
                # holes_bcp_val = pd_holes_df['bcp_val'].to_numpy() 
                # holes_dcp_val = pd_holes_df['dcp_val'].to_numpy() 
                # holes_pers_val = pd_holes_df['pers'].to_numpy() 
                # Because we use dtm, use euclidean distance between bcp coord and dcp coord as the persistence
                # holes_pers_val = np.linalg.norm(pd_holes_df[['bcp_y','bcp_x']].to_numpy() - pd_holes_df[['dcp_y','dcp_x']].to_numpy(), axis=1)
                holes_pers_val=np.sqrt(((np.stack([holes_bcp_y, holes_bcp_x], axis=-1) - np.stack([holes_dcp_y, holes_dcp_x], axis=-1))**2).sum(axis=1))
                pers_holes_df = pd.DataFrame(holes_pers_val)
                pers_hist_holes_df = pd.cut(pers_holes_df[0], hist_buckets_arr, right=True).value_counts(sort=False)
                pers_hist_holes_max = np.maximum(pers_hist_holes_max, pers_hist_holes_df.values)
                print('holes_pers_val', holes_pers_val)
                print('pers_hist_holes_df.values', pers_hist_holes_df.values)

        wi += 1
        sys.stdout.flush()

    pers_hist_cc_norm_factor = np.round((pers_hist_cc_max*1.25+2.51)/5)*5
    pers_hist_holes_norm_factor = np.round((pers_hist_holes_max*1.25+2.51)/5)*5

    with open(out_filepath, 'w+') as out_file:
        out_file.write('max cc:\n')
        out_file.write(str(pers_hist_cc_max.astype(int)))
        out_file.write('\n\n')
        out_file.write('norm factor cc:\n')
        out_file.write(str(pers_hist_cc_norm_factor.astype(int)))
        out_file.write('\n\n')
        out_file.write('max holes:\n')
        out_file.write(str(pers_hist_holes_max.astype(int)))
        out_file.write('\n\n')
        out_file.write('norm factor holes:\n')
        out_file.write(str(pers_hist_holes_norm_factor.astype(int)))
