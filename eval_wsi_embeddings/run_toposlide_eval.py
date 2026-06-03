import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import numpy as np
import h5py
import pandas as pd
import pickle

import glob
import os
# import traceback
import argparse

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

    args = parser.parse_args()

    patch_embedding_dir = args.patch_embedding_dir
    slide_root_output_dir = args.slide_root_output_dir
    patch_size_lv0 = args.patch_size_lv0
    checkpoint_path = args.checkpoint_path
    wsi_dim_filepath = args.wsi_dim_filepath

    if(patch_size_lv0<=0 ):
        raise Exception(f"patch_size_lv0={patch_size_lv0} must be greater than zero")

    os.makedirs(slide_root_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)

    config = TopoSlideConfig()
    config.vision_config.pos_encode_type = 'abs'
    print('config done')
    model = TopoSlide(config)
    print('model architecture created')

    state_dict2 = torch.load(checkpoint_path)
    state_dict = state_dict2['toposlide'] 

    # for key in state_dict.keys():
    #     print(key)
    print('stat_dict loaded')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    print('model loaded')
    print('model.device', model.device)


    # patch_feat_hdf_files = np.array(glob.glob(os.path.join(patch_embedding_dir, "*.h5")))
    # print('patch_feat_hdf_files', len(patch_feat_hdf_files))

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
        print('slide_name', slide_name)
        patch_embedding_hdf_file = h5py.File(patch_feat_filepath, 'r')
        features = patch_embedding_hdf_file['features'][:]
        coords = patch_embedding_hdf_file['coords'][:]

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

