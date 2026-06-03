
import os
import random
import glob
import random
import math
import random

import numpy as np
import h5py
import pandas as pd
from scipy import ndimage
from skimage.transform import rescale, resize

import torch
from torch.utils.data import Dataset




#########################################################################


class ClustersDataset(Dataset):
    def __init__(self, num_clusters, clustering_dir, clusters_all_stats_filepath, patch_embedding_dir, ignore_clusters_list, clusters_topo_dir, hist_buckets_arr_norm_factor_fixed_cc, hist_buckets_arr_norm_factor_fixed_holes, is_train, split_slide_names,wsi_dim_filepath, use_wsi_fraction=0.5):
        
        self.hist_buckets_arr = np.array([2, 5,10,20,30,40,50, math.inf])

        # 50x50 n patches = approx 50K x 50K @40x = 25Kx25K @20x --> diagonal = side x sqrt(2) = 50  x sqrt(2) = 70, radius = diagonal/2 = 35
        # 50x50 n patches = 35 radius max pers, pers=p -> comp_area = p x p, n_comp_p = (35X35)/(pXp)=(35/p)^2. At p=2 ->306, At p=5 ->49,...
        # 100x100 n patches = approx 100K x 100K @40x = 50Kx50K @20x --> diagonal = side x sqrt(2) = 100  x sqrt(2) = 140, radius = diagonal/2 = 70
        # 100x100 n patches = approx 100K x 100K @40x = 50Kx50K @20x --> diagonal = side = 50 (beacuase each pixel is exactly one point distance, imagine a grid), radius = diagonal/2 = 25
        # 100x100 n patches @20x --> diagonal = side = 100 (beacuase each pixel is exactly one point distance, imagine a grid), radius = diagonal/2 = 50

        # self.hist_buckets_arr_norm_factor_fixed_cc = np.array([135, 15,20,5,5,5,5]) # 1.25 * max in each bucket (across training data) [107,15,1,2,1,0,0] and take the nearest nonzero multiple of 5 
        # self.hist_buckets_arr_norm_factor_fixed_holes = np.array([90, 20,10,5,5,5,5]) # 1.25 * max in each bucket (across training data) [72,16,7,4,2,2,2] and take the nearest nonzero multiple of 5 

        
        self.tile_vis_size = 1
        self.kernel = np.ones((100,100))
        self.kernel_min = 100 # min is a region 10x10
        self.kernel_min = 50*50 # min is a region 50x50
        self.feature_dim = 768  # dim of CONCHv1.5 features
        self.min_group_tiles_proportion = 0.1
        self.use_wsi_fraction = use_wsi_fraction

        self.wsi_dim_df = pd.read_csv(wsi_dim_filepath)
        self.wsi_dim_arr = self.wsi_dim_df.to_numpy()        

        self.n_clusters = num_clusters

        self.groups_dict = {} # {<group_name>:[<group_indx>, list of cluster ids]}
        for ci in range(self.n_clusters):
            self.groups_dict[f"c{ci}"]=[ci,[ci]]
        gi = self.n_clusters
        self.group_ids_cat_single_list = np.arange(0,self.n_clusters)

        self.n_groups = num_clusters
        self.group_keys = []
        for group_name in self.groups_dict.keys():
            self.group_keys.append(group_name)
        self.group_keys = np.array(self.group_keys)


        self.num_clusters = num_clusters
        self.clustering_dir = clustering_dir
        self.clusters_all_stats_filepath = clusters_all_stats_filepath
        self.patch_embedding_dir = patch_embedding_dir
        self.clusters_topo_dir = clusters_topo_dir
        self.hist_buckets_arr_norm_factor_fixed_cc = hist_buckets_arr_norm_factor_fixed_cc  # 1.25 * max in each bucket (across training data) [107,15,1,2,1,0,0] and take the nearest nonzero multiple of 5 
        self.hist_buckets_arr_norm_factor_fixed_holes = hist_buckets_arr_norm_factor_fixed_holes  # 1.25 * max in each bucket (across training data) [72,16,7,4,2,2,2] and take the nearest nonzero multiple of 5 

        self.is_train = is_train
        self.split_slide_names = split_slide_names

        self.ignore_clusters_arr = np.array(ignore_clusters_list)
        self.include_cluster_ids = []
        for ci in range(num_clusters):
            if(ci in self.ignore_clusters_arr):
                continue
            self.include_cluster_ids.append(ci)
        self.include_cluster_ids = np.array(self.include_cluster_ids)

        self.patch_feat_hdf_files = np.array(glob.glob(os.path.join(self.patch_embedding_dir, "*.h5")))
        self.slide_names = np.array([os.path.basename(filepath).split('_')[0] for filepath in self.patch_feat_hdf_files])

        self.clusters_all_stats_df = pd.read_csv(self.clusters_all_stats_filepath)
        self.all_stats_slide_names = self.clusters_all_stats_df["slide_name"].to_numpy()
        self.all_stats_cluster_proportions  = self.clusters_all_stats_df[[f"proportion_{ci}" for ci in self.include_cluster_ids]].to_numpy()

        self.topo_files_hd = np.array(glob.glob(os.path.join(self.clusters_topo_dir, f"*_topo.hdf5")) )
        self.topo_slide_names = np.array([os.path.basename(filepath).split('_')[0] for filepath in self.topo_files_hd])

        self.patch_clustering_files = np.array(glob.glob(os.path.join(self.clustering_dir, f"*.csv")) )
        self.patch_clustering_slide_names = np.array([os.path.basename(filepath).split('_')[0] for filepath in self.patch_clustering_files])

        if(self.split_slide_names is not None):
            self.slide_names, indices0, indices1= np.intersect1d(self.split_slide_names, self.slide_names, return_indices=True)
            self.patch_feat_hdf_files = self.patch_feat_hdf_files[indices1] 
            self.split_slide_names = self.slide_names

        if(self.topo_slide_names is not None):
            self.slide_names, indices0, indices1= np.intersect1d(self.topo_slide_names, self.slide_names, return_indices=True)
            self.patch_feat_hdf_files = self.patch_feat_hdf_files[indices1] 
            self.topo_files_hd = self.topo_files_hd[indices0] 
            self.topo_slide_names = self.slide_names

        if(self.patch_clustering_slide_names is not None):
            self.slide_names, indices0, indices1= np.intersect1d(self.patch_clustering_slide_names, self.slide_names, return_indices=True)
            self.patch_feat_hdf_files = self.patch_feat_hdf_files[indices1] 
            if(self.topo_slide_names is not None):
                self.topo_files_hd = self.topo_files_hd[indices1] 
                self.topo_slide_names = self.slide_names
            self.patch_clustering_files = self.patch_clustering_files[indices0] 
            self.patch_clustering_slide_names = self.slide_names

        print('dataset len', len(self.patch_feat_hdf_files))

        self.slide_names=np.array(self.slide_names)

    def __len__(self):
        return len(self.patch_feat_hdf_files)
        
    def __getitem__(self, idx):

        print('idx', idx, self.slide_names[idx])

        # create dummy feats and coords grid (size is chosen arbitrarily here)
        patch_size_lv0 = torch.tensor(512)
        # grid_width = random.randint(2, 10)
        # grid_height = random.randint(2, 10)
        
        _, width, height,mag, pw = self.wsi_dim_arr[self.wsi_dim_arr[:,0]==self.slide_names[idx]][0]
        print('width, height', width, height)
        # patch_size_lv0 = pw
            
        slide_name = self.slide_names[idx]
        print('slide_name', slide_name)
        patch_feat_filepath = self.patch_feat_hdf_files[idx]
        topo_hd_filepath = self.topo_files_hd[idx]
        topo_meta_filepath = topo_hd_filepath.replace(".hdf5", "_meta.csv")

        patch_clustering_filepath = self.patch_clustering_files[idx]

        ##############################################################
        # get negative sample
        ##############################################################
        while(True):
            idx_neg = random.choice(np.arange(len(self.slide_names)))
            if(idx_neg != idx):
                try:
                    slide_name_neg = self.slide_names[idx_neg]
                    print('slide_name_neg', slide_name_neg)
                    _, width_neg, height_neg,mag_neg, pw_neg = self.wsi_dim_arr[self.wsi_dim_arr[:,0]==self.slide_names[idx_neg]][0]
                    # patch_size_lv0_neg = pw_neg
                    patch_feat_filepath_neg = self.patch_feat_hdf_files[idx_neg]
                    topo_hd_filepath_neg = self.topo_files_hd[idx_neg]
                    patch_clustering_filepath_neg = self.patch_clustering_files[idx_neg]
                    patch_embedding_hdf_file_neg = h5py.File(patch_feat_filepath_neg, 'r')
                    embedding_arr_neg = patch_embedding_hdf_file_neg['features'][:]
                    topo_hdf_file_neg = h5py.File(topo_hd_filepath_neg, 'r')
                    fg_mask_neg = topo_hdf_file_neg['fg_mask.png'][:].astype(float)
                    fg_mask_neg /= 255
                    cluster_id_map_neg = topo_hdf_file_neg['cluster_id_map.npy'][:]
                    break
                except:
                    pass
        

        clustering_df_neg = pd.read_csv(patch_clustering_filepath_neg)
        if(self.ignore_clusters_arr is not None):
            clustering_df_neg = clustering_df_neg[~clustering_df_neg['cluster_id'].isin(self.ignore_clusters_arr)] 
        tiles_coord_x_neg = clustering_df_neg["coord_x"].to_numpy()
        tiles_coord_y_neg = clustering_df_neg["coord_y"].to_numpy()
        tiles_cluster_id_neg = clustering_df_neg["cluster_id"].to_numpy()
        tiles_cluster_dist_neg = clustering_df_neg["cluster_dist"].to_numpy()
        tile_size_neg = pw_neg

        scale_neg = self.tile_vis_size/tile_size_neg
        scale_fg_neg = fg_mask_neg.shape[0]/height_neg

        fg_mask_neg = rescale(fg_mask_neg.astype(int), scale_neg/scale_fg_neg, order=0, preserve_range=True)
        cluster_id_map_neg = resize(cluster_id_map_neg.astype(int), fg_mask_neg.shape, order=0)

        coord_arr_neg = patch_embedding_hdf_file_neg['coords'][:]
        coord_arr_scaled_neg = np.round(coord_arr_neg*scale_neg).astype(int)
        coord_arr_scaled_x_neg = coord_arr_scaled_neg[:, 0].clip(0,fg_mask_neg.shape[1]-1)
        coord_arr_scaled_y_neg = coord_arr_scaled_neg[:, 1].clip(0,fg_mask_neg.shape[0]-1)
        cluster_ids_arr_neg = cluster_id_map_neg[(coord_arr_scaled_y_neg, coord_arr_scaled_x_neg)]

        ##############################################################

        # Get the scale
        clustering_df = pd.read_csv(patch_clustering_filepath)
        if(self.ignore_clusters_arr is not None):
            clustering_df = clustering_df[~clustering_df['cluster_id'].isin(self.ignore_clusters_arr)] 
        tiles_coord_x = clustering_df["coord_x"].to_numpy()
        tiles_coord_y = clustering_df["coord_y"].to_numpy()
        tiles_cluster_id = clustering_df["cluster_id"].to_numpy()
        tiles_cluster_dist = clustering_df["cluster_dist"].to_numpy()
        tile_size = pw
        scale = self.tile_vis_size/tile_size
        scale512 = 512/tile_size

        
        # read feat and coord from patch_feat_filepath 
        # Get the embeddings and scale the patch coordinates
        patch_embedding_hdf_file = h5py.File(patch_feat_filepath, 'r')
        embedding_arr = patch_embedding_hdf_file['features'][:]
        coord_arr = patch_embedding_hdf_file['coords'][:]
        coord_arr_scaled = np.round(coord_arr*scale).astype(int)
        coord_arr_scaled_x = coord_arr_scaled[:, 0]
        coord_arr_scaled_y = coord_arr_scaled[:, 1]

        # Load topo file
        topo_hdf_file = h5py.File(topo_hd_filepath, 'r')
        topo_meta_df = pd.read_csv(topo_meta_filepath)
        topo_filenames_list = topo_meta_df["key_name"].to_numpy()

        # read fg and cluster id map from topo
        # Get  Fg mask
        fg_mask = topo_hdf_file['fg_mask.png'][:].astype(float)
        fg_mask /= 255

        scale_fg = fg_mask.shape[0]/height

        fg_mask = rescale(fg_mask.astype(int), scale/scale_fg, order=0, preserve_range=True)
        fg_rescale_scale = scale/scale_fg

        coord_arr_scaled_x = coord_arr_scaled_x.clip(0,fg_mask.shape[1]-1)
        coord_arr_scaled_y = coord_arr_scaled_y.clip(0,fg_mask.shape[0]-1)

        # Get cluster_id_map 
        cluster_id_map = topo_hdf_file['cluster_id_map.npy'][:]
        cluster_id_map = resize(cluster_id_map.astype(int), fg_mask.shape, order=0)

        # Get ROI
        if(self.is_train and random.random() <= self.use_wsi_fraction):
            # convolve fg with 100x100, get value range, select randomly from range between 100 and max, randomly select from map a center point that has that vlaue
            fg_conv_arr = ndimage.convolve(fg_mask, self.kernel, mode='constant', cval=0.0)
            fg_conv_arr = fg_conv_arr * fg_mask
            if((fg_conv_arr>=self.kernel_min).sum()==0):
                sel_conv_val = fg_conv_arr.max()
            else:
                min_conv_range = int(fg_conv_arr[fg_conv_arr>=self.kernel_min].min())
                max_conv_range = int(fg_conv_arr.max())
                sel_conv_val = random.randint(min_conv_range, max_conv_range)

            sel_conv_val_min_sqrt = int(math.sqrt(sel_conv_val))
            # sel_conv_val_max_sqrt = sel_conv_val_min + 1
            sel_conv_val_min = sel_conv_val_min_sqrt**2
            # sel_conv_val_max = sel_conv_val_max_sqrt**2
            fg_conv_min_arr = ndimage.convolve(fg_mask, np.ones((sel_conv_val_min_sqrt,sel_conv_val_min_sqrt)), mode='constant', cval=0.0)>=sel_conv_val_min**2
            # fg_conv_max_arr = ndimage.convolve(fg_mask, np.ones((sel_conv_val_max_sqrt,sel_conv_val_max_sqrt)), mode='constant', cval=0.0)>=sel_conv_val
            while(fg_conv_min_arr.sum()==0):
                sel_conv_val_min_sqrt -= 1
                fg_conv_min_arr = ndimage.convolve(fg_mask, np.ones((sel_conv_val_min_sqrt,sel_conv_val_min_sqrt)), mode='constant', cval=0.0)>=sel_conv_val_min_sqrt**2
            (candidate_center_y, candidate_center_x) = np.where(fg_conv_min_arr>0)
            sel_center_idx = random.randint(0, len(candidate_center_y)-1)
            cy = candidate_center_y[sel_center_idx]
            cx = candidate_center_x[sel_center_idx]
            roi_min_y = max(0, cy - sel_conv_val_min_sqrt//2)
            roi_min_x = max(0, cx - sel_conv_val_min_sqrt//2)
            roi_max_y = min(fg_mask.shape[0], roi_min_y + sel_conv_val_min_sqrt)
            roi_max_x = min(fg_mask.shape[1], roi_min_x + sel_conv_val_min_sqrt)
            roi_side_len = sel_conv_val_min_sqrt
            roi_mask = np.zeros(fg_mask.shape)
            xv, yv = np.meshgrid(np.arange(roi_min_x, roi_max_x), np.arange(roi_min_y, roi_max_y))
            roi_mask[(yv, xv)] = 1
            roi_mask_fg = roi_mask*fg_mask
            if(roi_mask_fg.sum() <= 1):
                roi_mask = fg_mask.copy()
                roi_mask_fg = fg_mask.copy()
        else:
            roi_mask_fg = fg_mask
            roi_mask = fg_mask

        (roi_fg_coord_y, roi_fg_coord_x) = np.where(roi_mask_fg>0)
        (roi_coord_y, roi_coord_x) = np.where(roi_mask>0)

        # Get ROI patch embeddings and coordinates
        embedding_im = np.zeros((fg_mask.shape[0], fg_mask.shape[1], self.feature_dim ))
        embedding_im[(coord_arr_scaled_y, coord_arr_scaled_x)] = embedding_arr
        roi_patch_embeddings = embedding_im[(roi_coord_y, roi_coord_x)]
        roi_coord_y_512 = roi_coord_y*512
        roi_coord_x_512 = roi_coord_x*512

        # Get ROI cluster ids and proportions
        roi_patch_cluster_ids = cluster_id_map[(roi_coord_y, roi_coord_x)].astype(int)
        roi_cluster_ids, roi_cluster_counts = np.unique(roi_patch_cluster_ids, return_counts=True)
        roi_cluster_ids_all = np.arange(self.num_clusters)
        roi_cluster_counts_all = np.zeros(self.num_clusters)
        roi_cluster_counts_all[roi_cluster_ids] = roi_cluster_counts
        roi_cluster_proportion_all = roi_cluster_counts_all/(roi_mask_fg.sum())


        # get topo in ROI
        
        cp_maps_cc = np.zeros((fg_mask.shape[0], fg_mask.shape[1], self.n_groups))
        cp_maps_holes = np.zeros((fg_mask.shape[0], fg_mask.shape[1], self.n_groups))
        
        pers_hist_cc = np.zeros((self.n_groups, len(self.hist_buckets_arr)-1))
        pers_hist_holes = np.zeros((self.n_groups, len(self.hist_buckets_arr)-1))
        pers_hist_cc_norm = np.zeros((self.n_groups, len(self.hist_buckets_arr)-1))
        pers_hist_holes_norm = np.zeros((self.n_groups, len(self.hist_buckets_arr)-1))

        group_pnt_maps = np.zeros((fg_mask.shape[0], fg_mask.shape[1], self.n_groups))


        for group_name, val in self.groups_dict.items():
            gi = val[0]
            cluster_list = val[1]
            
            group_map = topo_hdf_file[f'pnts_g{gi}_{group_name}.png'][:].astype(float)
            group_map = resize(group_map.astype(int), fg_mask.shape, order=0)
            group_pnt_maps[:,:,gi] = group_map  * roi_mask_fg 

            pnt_cloud_name = f"pnts_cc_{group_name}_g{gi}"
            filename_pattern = f"{pnt_cloud_name}_pnts_cc_exclude_bg_topo_cps_original_coord.csv"
            r=pd.Series(topo_filenames_list).str.match(filename_pattern)
            selected_keys = topo_filenames_list[r]
            if(len(selected_keys)>0):
                csv_ds = topo_hdf_file[filename_pattern]
                pd_cc_df = pd.DataFrame(csv_ds['data'][:], index=csv_ds['index'][:], columns=csv_ds['columns'][:].astype(str))
                cc_bcp_y = (pd_cc_df['bcp_y'].to_numpy()*scale).clip(0,fg_mask.shape[0]-1)
                cc_bcp_x = (pd_cc_df['bcp_x'].to_numpy()*scale).clip(0,fg_mask.shape[1]-1) 
                cc_dcp_y = (pd_cc_df['dcp_y'].to_numpy()*scale).clip(0,fg_mask.shape[0]-1) 
                cc_dcp_x = (pd_cc_df['dcp_x'].to_numpy()*scale).clip(0,fg_mask.shape[1]-1) 
                cc_bcp_val = pd_cc_df['bcp_val'].to_numpy() 
                cc_dcp_val = pd_cc_df['dcp_val'].to_numpy() 
                cc_pers_val = pd_cc_df['pers'].to_numpy() 
                # Because we use dtm, use euclidean distance between bcp coord and dcp coord as the persistence
                # cc_pers_val = np.linalg.norm(pd_cc_df[['bcp_y','bcp_x']].to_numpy() - pd_cc_df[['dcp_y','dcp_x']].to_numpy(), axis=1)
                cc_pers_val=np.sqrt(((np.stack([cc_bcp_y, cc_bcp_x], axis=-1) - np.stack([cc_dcp_y, cc_dcp_x], axis=-1))**2).sum(axis=1))
                cp_maps_cc[(cc_bcp_y.astype(int), cc_bcp_x.astype(int), np.ones(cc_pers_val.shape[0],dtype=int)*gi)]=cc_pers_val
                cp_maps_cc[:,:,gi] = cp_maps_cc[:,:,gi]* roi_mask_fg

                pers_cc_df = pd.DataFrame(cc_pers_val)
                pers_hist_cc_df = pd.cut(pers_cc_df[0], self.hist_buckets_arr, right=True).value_counts(sort=False)
                pers_hist_cc[gi] = pers_hist_cc_df.values
                pers_hist_cc_norm[gi] = np.divide(pers_hist_cc_df.values, self.hist_buckets_arr_norm_factor_fixed_cc)
            

            pnt_cloud_name = f"pnts_holes_{group_name}_g{gi}"
            filename_pattern = f"{pnt_cloud_name}_pnts_holes_exclude_bg_filter_hull_topo_cps_original_coord.csv"
            r=pd.Series(topo_filenames_list).str.match(filename_pattern)
            selected_keys = topo_filenames_list[r]
            if(len(selected_keys)>0):
                csv_ds = topo_hdf_file[f"{pnt_cloud_name}_pnts_holes_exclude_bg_filter_hull_topo_cps_original_coord.csv"]
                pd_holes_df = pd.DataFrame(csv_ds['data'][:], index=csv_ds['index'][:], columns=csv_ds['columns'][:].astype(str))
                holes_bcp_y = (pd_holes_df['bcp_y'].to_numpy()*scale).clip(0,fg_mask.shape[0]-1)
                holes_bcp_x = (pd_holes_df['bcp_x'].to_numpy()*scale).clip(0,fg_mask.shape[1]-1) 
                holes_dcp_y = (pd_holes_df['dcp_y'].to_numpy()*scale).clip(0,fg_mask.shape[0]-1) 
                holes_dcp_x = (pd_holes_df['dcp_x'].to_numpy()*scale).clip(0,fg_mask.shape[1]-1) 
                holes_bcp_val = pd_holes_df['bcp_val'].to_numpy() 
                holes_dcp_val = pd_holes_df['dcp_val'].to_numpy() 
                holes_pers_val = pd_holes_df['pers'].to_numpy() 
                # Because we use dtm, use euclidean distance between bcp coord and dcp coord as the persistence
                # holes_pers_val = np.linalg.norm(pd_holes_df[['bcp_y','bcp_x']].to_numpy() - pd_holes_df[['dcp_y','dcp_x']].to_numpy(), axis=1)
                holes_pers_val=np.sqrt(((np.stack([holes_bcp_y, holes_bcp_x], axis=-1) - np.stack([holes_dcp_y, holes_dcp_x], axis=-1))**2).sum(axis=1))
                if(len(holes_pers_val)==1):
                    cp_maps_holes[(holes_bcp_y.astype(int), holes_bcp_x.astype(int), gi)]=holes_pers_val
                else:
                    try:
                        cp_maps_holes[(holes_bcp_y.astype(int), holes_bcp_x.astype(int), np.ones(holes_pers_val.shape[0],dtype=int)*gi)]=holes_pers_val
                    except Exception as e:
                        print(e)
                        print('cp_maps_holes[(holes_bcp_y.astype(int), holes_bcp_x.astype(int), np.ones(holes_pers_val.shape[0],dtype=int)*gi)]=holes_pers_val')
                        print('cp_maps_holes', cp_maps_holes.shape)
                        print('fg_mask', fg_mask.shape)
                        print('holes_bcp_y', holes_bcp_y.max())
                        print('holes_bcp_x', holes_bcp_x.max())
                        print('holes_dcp_y', holes_dcp_y.max())
                        print('holes_dcp_x', holes_dcp_x.max())
                        raise
                cp_maps_holes[:,:,gi] = cp_maps_holes[:,:,gi]* roi_mask_fg

                pers_holes_df = pd.DataFrame(holes_pers_val)
                pers_hist_holes_df = pd.cut(pers_holes_df[0], self.hist_buckets_arr, right=True).value_counts(sort=False)
                pers_hist_holes[gi] = pers_hist_holes_df.values
                pers_hist_holes_norm[gi] = np.divide(pers_hist_holes_df.values, self.hist_buckets_arr_norm_factor_fixed_holes)


         
        # self.max_pers_cc = np.maximum(self.max_pers_cc, pers_hist_cc)
        # self.max_pers_holes = np.maximum(self.max_pers_holes, pers_hist_holes)

        # normalize group_pnt_maps
        group_pnt_maps /=255
        group_pnt_prop = group_pnt_maps.sum(axis=(0,1))/(roi_mask_fg.sum())
        group_pnt_sum = group_pnt_maps.sum(axis=(0,1))
        # normalize pers_maps
        cp_maps_cc_norm = cp_maps_cc / self.hist_buckets_arr[-2]
        cp_maps_holes_norm = cp_maps_holes / self.hist_buckets_arr[-2]
        
        selected_groups_ids = []
        selected_groups_sample_emb = []
        selected_empty_group = False
        group_keys_permuted = np.random.permutation(self.group_keys)
        for group_name in group_keys_permuted:
            gi = self.groups_dict[group_name][0]
            cluster_list = val[1]
            if(gi in self.group_ids_cat_single_list):
                if(group_pnt_prop[gi]>=self.min_group_tiles_proportion):
                    selected_patch_indices = np.random.choice(int(group_pnt_maps[:,:,gi].sum()), size=4)
                    selected_groups_ids.append(np.ones(len(selected_patch_indices))*gi)
                    selected_groups_sample_emb.append(embedding_im[group_pnt_maps[:,:,gi]>0][selected_patch_indices])
                elif(not selected_empty_group and group_pnt_sum[gi] > 0 and group_pnt_sum[gi] <6):
                    selected_patch_indices = np.arange(min(group_pnt_sum[gi],4)).astype(int)
                    selected_groups_ids.append(np.ones(len(selected_patch_indices))*gi)
                    selected_groups_sample_emb.append(embedding_im[group_pnt_maps[:,:,gi]>0][selected_patch_indices])
                    selected_empty_group = True
            
        selected_groups_ids = np.concatenate(selected_groups_ids, axis=0).astype(int)
        selected_groups_sample_emb = np.concatenate(selected_groups_sample_emb, axis=0)

        feats = torch.tensor(embedding_im[roi_mask_fg>0], dtype=torch.bfloat16)
        coords = torch.tensor(np.stack((roi_fg_coord_y, roi_fg_coord_x), axis=1), dtype=torch.int64)*patch_size_lv0
        patch_size_lv0=torch.tensor(patch_size_lv0)
        roi_cluster_proportion_all = torch.tensor(roi_cluster_proportion_all, dtype=torch.bfloat16)
        # roi_group_proportion_all = torch.tensor(group_pnt_prop, dtype=torch.bfloat16)
        pers_hist_cc = torch.tensor(pers_hist_cc, dtype=torch.bfloat16)
        pers_hist_cc_norm = torch.tensor(pers_hist_cc_norm, dtype=torch.bfloat16)
        pers_hist_holes = torch.tensor(pers_hist_holes, dtype=torch.bfloat16)
        pers_hist_holes_norm = torch.tensor(pers_hist_holes_norm, dtype=torch.bfloat16)

        selected_groups_ids = torch.tensor(selected_groups_ids, dtype=torch.int)
        selected_groups_sample_emb = torch.tensor(selected_groups_sample_emb, dtype=torch.bfloat16)

        token_cp_pers_cc = torch.tensor(cp_maps_cc[roi_mask_fg>0], dtype=torch.bfloat16)
        token_cp_pers_holes = torch.tensor(cp_maps_holes[roi_mask_fg>0], dtype=torch.bfloat16)
        token_cp_pers_cc_norm = torch.tensor(cp_maps_cc_norm[roi_mask_fg>0], dtype=torch.bfloat16)
        token_cp_pers_holes_norm = torch.tensor(cp_maps_holes_norm[roi_mask_fg>0], dtype=torch.bfloat16)

        roi_patch_cluster_ids = torch.tensor(roi_patch_cluster_ids, dtype=torch.int)
        roi_patch_embeddings = torch.tensor(roi_patch_embeddings, dtype=torch.bfloat16)
        cluster_ids_arr_neg = torch.tensor(cluster_ids_arr_neg, dtype=torch.int)
        embedding_arr_neg = torch.tensor(embedding_arr_neg, dtype=torch.bfloat16)


        return feats, coords, patch_size_lv0 \
            , roi_cluster_proportion_all \
            , pers_hist_cc_norm, pers_hist_holes_norm \
            , selected_groups_ids, selected_groups_sample_emb \
            , token_cp_pers_cc, token_cp_pers_holes, token_cp_pers_cc_norm, token_cp_pers_holes_norm \
            , roi_patch_cluster_ids, roi_patch_embeddings, cluster_ids_arr_neg, embedding_arr_neg