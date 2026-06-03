from re import split
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
import json
import shutil
import io
import math

from tqdm import tqdm
import torch
import torch.nn.functional as F

from model.build_toposlide import TopoSlide
from model.configuration import TopoSlideConfig
from dataloader.clusters_dataset2 import ClustersDataset




def init_weights(my_module, conv_init='normal'):
    for layer in my_module:
        if(isinstance(layer, torch.nn.ConvTranspose2d) 
            or isinstance(layer, torch.nn.Conv2d)
            or isinstance(layer, torch.nn.Linear)
            ):
            if(conv_init == 'normal'):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1) ; # config option: mlp.weight.data.normal_(mean=0.0, std=0.01) default is std = 1
            elif(conv_init == 'xavier_uniform'):
                torch.nn.init.xavier_uniform_(layer.weight) ;
            elif(conv_init == 'xavier_normal'):
                torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
            elif(conv_init == 'he'):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
        if(isinstance(layer, torch.nn.Linear)):
            layer.bias.data.zero_()


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    """Copied from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/scheduler.py
    """
    def _warmup_lr(base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length
    
    def _assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = new_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = new_lr
    
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Train TopoSlide")    
    parser.add_argument("--name", default='toposlide', type=str)

    # # Training Arguments
    parser.add_argument("--checkpoints_root_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/checkpoints")
    parser.add_argument("--checkpoints_folder_name", type=str, default="tcga_luad")
    parser.add_argument("--model_param_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_warmup", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--model_save_freq", type=int, default=5)
    parser.add_argument("--start_save_epoch", type=int, default=50)
    

    # # Loss Weights
    parser.add_argument("--cond2_pers_hist_norm_loss_weight", type=float, default=0.5)
    parser.add_argument("--cond2_cluster_prop_loss_weight", type=float, default=1)
    parser.add_argument("--cond2_pers_hist_norm_cumsum_loss_weight", type=float, default=0.5)
    parser.add_argument("--cond2_cluster_prop_cumsum_loss_weight", type=float, default=1)
    parser.add_argument("--token_cp_pers_loss_weight", type=float, default=0.01)
    parser.add_argument("--token_cp_pers_norm_loss_weight", type=float, default=0.5)
    parser.add_argument("--token_cp_pers_binary_loss_weight", type=float, default=1)
    parser.add_argument("--token_similarity_loss_weight", type=float, default=0.25)

    # # Dataset Arguments
    parser.add_argument("--patch_size_lv0", type=int, default=512)
    parser.add_argument("--tile_embedding_size", type=int, default=768)    
    parser.add_argument("--num_clusters", type=int, default=16)
    parser.add_argument("--wsi_dim_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv")
    parser.add_argument("--patch_embedding_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding")
    parser.add_argument("--clustering_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans")
    parser.add_argument("--clusters_all_stats_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/clustering_stats/cluster_stats.csv")
    parser.add_argument("--clusters_topo_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo")
    parser.add_argument("--ignore_cluster_ids", type=int, nargs="+", default=[1,8,0])
    parser.add_argument("--train_split_filepath", type=str, default=f"/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/train_split/train_1.csv")
    parser.add_argument("--val_split_filepath", type=str, default=f"/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/train_split/test_1.csv")
    parser.add_argument("--hist_buckets_arr_norm_factor_fixed_cc", type=int, nargs="+", default=[140, 25,10,5,5,5,5]) # for TCGA BRCA: [135, 15,20,5,5,5,5]
    parser.add_argument("--hist_buckets_arr_norm_factor_fixed_holes", type=int, nargs="+", default=[140, 25,10,5,5,5,5]) # for TCGA BRCA: [90, 20,10,5,5,5,5]
    
    args = parser.parse_args()


    # # Read Training Arguments
    checkpoints_root_dir = args.checkpoints_root_dir
    checkpoints_folder_name = args.checkpoints_folder_name
    checkpoints_save_path   = os.path.join(checkpoints_root_dir, checkpoints_folder_name)
    model_param_path = args.model_param_path
    warmup_epochs = args.warmup_epochs
    start_epoch = args.start_epoch
    model_save_freq = args.model_save_freq
    start_save_epoch = args.start_save_epoch

    # # Read Loss Weights
    cond2_pers_hist_norm_loss_weight = args.cond2_pers_hist_norm_loss_weight
    cond2_cluster_prop_loss_weight = args.cond2_cluster_prop_loss_weight
    cond2_pers_hist_norm_cumsum_loss_weight = args.cond2_pers_hist_norm_cumsum_loss_weight
    cond2_cluster_prop_cumsum_loss_weight = args.cond2_cluster_prop_cumsum_loss_weight
    token_cp_pers_loss_weight = args.token_cp_pers_loss_weight
    token_cp_pers_norm_loss_weight = args.token_cp_pers_norm_loss_weight
    token_cp_pers_binary_loss_weight = args.token_cp_pers_binary_loss_weight    
    token_similarity_loss_weight = args.token_similarity_loss_weight

    # # Read Dataset Arguments
    patch_size_lv0 = args.patch_size_lv0
    num_clusters = args.num_clusters
    wsi_dim_filepath = args.wsi_dim_filepath
    patch_embedding_dir = args.patch_embedding_dir
    clustering_dir = args.clustering_dir
    clusters_all_stats_filepath = args.clusters_all_stats_filepath
    clusters_topo_dir = args.clusters_topo_dir
    ignore_clusters_list = np.array(list(set(args.ignore_cluster_ids)))
    train_split_filepath = args.train_split_filepath
    val_split_filepath = args.val_split_filepath 
    tile_embedding_size = args.tile_embedding_size 
    slide_embedding_size = tile_embedding_size
    hist_buckets_arr_norm_factor_fixed_cc = np.array(list(args.hist_buckets_arr_norm_factor_fixed_cc))
    hist_buckets_arr_norm_factor_fixed_holes = np.array(list(args.hist_buckets_arr_norm_factor_fixed_holes))

    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    min_total_loss_val = math.inf
    min_val_loss = math.inf
    ema_alpha = 0.2
    grad_smoothed_min = math.inf
    patience = 50
    best_epoch = -1

    os.makedirs(checkpoints_save_path, exist_ok=True)

    # Save command line arguments    
    with open(os.path.join(checkpoints_save_path, f'commandline_args_e{start_epoch}.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)

    # Set up data loaders
    train_slide_names = pd.read_csv(train_split_filepath)['slide_ids'].values
    val_slide_names = pd.read_csv(val_split_filepath)['slide_ids'].values

    train_dataset = ClustersDataset(num_clusters, clustering_dir, clusters_all_stats_filepath, patch_embedding_dir, ignore_clusters_list, clusters_topo_dir, hist_buckets_arr_norm_factor_fixed_cc, hist_buckets_arr_norm_factor_fixed_holes, is_train=True, split_slide_names=train_slide_names, wsi_dim_filepath=wsi_dim_filepath)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)

    val_dataset = ClustersDataset(num_clusters, clustering_dir, clusters_all_stats_filepath, patch_embedding_dir, ignore_clusters_list, clusters_topo_dir, hist_buckets_arr_norm_factor_fixed_cc, hist_buckets_arr_norm_factor_fixed_holes, is_train=False, split_slide_names=val_slide_names, wsi_dim_filepath=wsi_dim_filepath)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)
    

    # Load model
    config = TopoSlideConfig()
    config.vision_config.pos_encode_type = 'abs'
    config.vision_config.return_all_tokens_before_pooling = True
    print('config done')
    model = TopoSlide(config)
    print('model architecture created')

    if(model_param_path is not None):
        print('model_param_path', model_param_path)
        state_dict2 = torch.load(model_param_path)
        model.load_state_dict(state_dict2['toposlide'], strict=True)

    model = model.to(device)

    print('main model loaded')
    print('model.device', model.device)


    #############################
    layer_indx = 0
    mlp_project_wsi_embed1 = torch.nn.Sequential()
    layer_indx += 1
    mlp_project_wsi_embed1.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, slide_embedding_size))
    mlp_project_wsi_embed1 = mlp_project_wsi_embed1.to(device)

    layer_indx = 0
    mlp_project_patch_embed1 = torch.nn.Sequential()
    layer_indx += 1
    mlp_project_patch_embed1.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(tile_embedding_size, tile_embedding_size))
    mlp_project_patch_embed1 = mlp_project_patch_embed1.to(device)


    #############################
    # concatenate the wsi emb and the conditional patch embedding, and apply mlp that encodes into smaller dim
    layer_indx = 0
    mlp_cond2_prop_clusters_mha_enc = torch.nn.Sequential()
    layer_indx += 1
    mlp_cond2_prop_clusters_mha_enc.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size+tile_embedding_size, (slide_embedding_size+tile_embedding_size)//8))
    mlp_cond2_prop_clusters_mha_enc = mlp_cond2_prop_clusters_mha_enc.to(device)

    # use the encoding of the concatenated wsi and tile to create n channel wise attention maps with the slide dim
    mlp_cond2_prop_clusters_n_heads_att = 1
    mlp_cond2_prop_clusters_mha = torch.nn.ModuleList([torch.nn.Linear((slide_embedding_size+tile_embedding_size)//8, slide_embedding_size) for _ in range(mlp_cond2_prop_clusters_n_heads_att)])
    mlp_cond2_prop_clusters_mha = mlp_cond2_prop_clusters_mha.to(device)
    
    # project the wsi for the attention value
    # the projected wsi will be multiplied by the channel wise attention map 
    mlp_cond2_prop_clusters_wsi_mval = torch.nn.ModuleList([torch.nn.Linear(slide_embedding_size, slide_embedding_size) for _ in range(mlp_cond2_prop_clusters_n_heads_att)])
    mlp_cond2_prop_clusters_wsi_mval = mlp_cond2_prop_clusters_wsi_mval.to(device)
    
    # apply an mlp for each attention result, if more than one then concat later and apply mlp on combination to finally generate the target prediction
    mlp_cond2_prop_clusters_mha_dec_list = []
    for i in range(mlp_cond2_prop_clusters_n_heads_att):
        layer_indx = 0
        mlp_cond2_prop_clusters_mha_dec = torch.nn.Sequential()
        layer_indx += 1
        mlp_cond2_prop_clusters_mha_dec.add_module(f'mlp{i}_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, 1))
        mlp_cond2_prop_clusters_mha_dec_list.append(mlp_cond2_prop_clusters_mha_dec)
    mlp_cond2_prop_clusters_mha_dec = torch.nn.ModuleList(mlp_cond2_prop_clusters_mha_dec_list).to(device)


    #############################
    # concatenate the wsi emb and the conditional patch embedding, and apply mlp that encodes into smaller dim
    layer_indx = 0
    mlp_cond2_pers_hist_cc_norm_mha_enc = torch.nn.Sequential()
    layer_indx += 1
    mlp_cond2_pers_hist_cc_norm_mha_enc.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size+tile_embedding_size, (slide_embedding_size+tile_embedding_size)//8))
    mlp_cond2_pers_hist_cc_norm_mha_enc = mlp_cond2_pers_hist_cc_norm_mha_enc.to(device)

    # use the encoding of the concatenated wsi and tile to create n channel wise attention maps with the slide dim
    mlp_cond2_pers_hist_cc_norm_n_heads_att = 8
    mlp_cond2_pers_hist_cc_norm_mha = torch.nn.ModuleList([torch.nn.Linear((slide_embedding_size+tile_embedding_size)//8, slide_embedding_size) for _ in range(mlp_cond2_pers_hist_cc_norm_n_heads_att)])
    mlp_cond2_pers_hist_cc_norm_mha = mlp_cond2_pers_hist_cc_norm_mha.to(device)
    
    # project the wsi for the attention value
    # the projected wsi will be multiplied by the channel wise attention map 
    mlp_cond2_pers_hist_cc_norm_wsi_mval = torch.nn.ModuleList([torch.nn.Linear(slide_embedding_size, slide_embedding_size) for _ in range(mlp_cond2_pers_hist_cc_norm_n_heads_att)])
    mlp_cond2_pers_hist_cc_norm_wsi_mval = mlp_cond2_pers_hist_cc_norm_wsi_mval.to(device)
    
    # apply an mlp for each attention result, if more than one then concat later and apply mlp on combination to finally generate the target prediction
    mlp_cond2_pers_hist_cc_norm_mha_dec_list = []
    for i in range(mlp_cond2_pers_hist_cc_norm_n_heads_att):
        layer_indx = 0
        mlp_cond2_pers_hist_cc_norm_mha_dec = torch.nn.Sequential()
        layer_indx += 1
        mlp_cond2_pers_hist_cc_norm_mha_dec.add_module(f'mlp{i}_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, 16))
        mlp_cond2_pers_hist_cc_norm_mha_dec_list.append(mlp_cond2_pers_hist_cc_norm_mha_dec)
    mlp_cond2_pers_hist_cc_norm_mha_dec = torch.nn.ModuleList(mlp_cond2_pers_hist_cc_norm_mha_dec_list).to(device)

    layer_indx = 0
    mlp_cond2_pers_hist_cc_norm_mha_dec_cat = torch.nn.Sequential()
    layer_indx += 1
    mlp_cond2_pers_hist_cc_norm_mha_dec_cat.add_module(f'mlp{i}_{layer_indx}_linear', torch.nn.Linear(16*mlp_cond2_pers_hist_cc_norm_n_heads_att, len(train_dataset.hist_buckets_arr)-1))
    mlp_cond2_pers_hist_cc_norm_mha_dec_cat = mlp_cond2_pers_hist_cc_norm_mha_dec_cat.to(device)



    #############################
    # concatenate the wsi emb and the conditional patch embedding, and apply mlp that encodes into smaller dim
    layer_indx = 0
    mlp_cond2_pers_hist_loop_norm_mha_enc = torch.nn.Sequential()
    layer_indx += 1
    mlp_cond2_pers_hist_loop_norm_mha_enc.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size+tile_embedding_size, (slide_embedding_size+tile_embedding_size)//8))
    mlp_cond2_pers_hist_loop_norm_mha_enc = mlp_cond2_pers_hist_loop_norm_mha_enc.to(device)

    # use the encoding of the concatenated wsi and tile to create n channel wise attention maps with the slide dim
    mlp_cond2_pers_hist_loop_norm_n_heads_att = 8
    mlp_cond2_pers_hist_loop_norm_mha = torch.nn.ModuleList([torch.nn.Linear((slide_embedding_size+tile_embedding_size)//8, slide_embedding_size) for _ in range(mlp_cond2_pers_hist_loop_norm_n_heads_att)])
    mlp_cond2_pers_hist_loop_norm_mha = mlp_cond2_pers_hist_loop_norm_mha.to(device)
    
    # project the wsi for the attention value
    # the projected wsi will be multiplied by the channel wise attention map 
    mlp_cond2_pers_hist_loop_norm_wsi_mval = torch.nn.ModuleList([torch.nn.Linear(slide_embedding_size, slide_embedding_size) for _ in range(mlp_cond2_pers_hist_loop_norm_n_heads_att)])
    mlp_cond2_pers_hist_loop_norm_wsi_mval = mlp_cond2_pers_hist_loop_norm_wsi_mval.to(device)
    
    # apply an mlp for each attention result, if more than one then concat later and apply mlp on combination to finally generate the target prediction
    mlp_cond2_pers_hist_loop_norm_mha_dec_list = []
    for i in range(mlp_cond2_pers_hist_loop_norm_n_heads_att):
        layer_indx = 0
        mlp_cond2_pers_hist_loop_norm_mha_dec = torch.nn.Sequential()
        layer_indx += 1
        mlp_cond2_pers_hist_loop_norm_mha_dec.add_module(f'mlp{i}_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, 16))
        mlp_cond2_pers_hist_loop_norm_mha_dec_list.append(mlp_cond2_pers_hist_loop_norm_mha_dec)
    mlp_cond2_pers_hist_loop_norm_mha_dec = torch.nn.ModuleList(mlp_cond2_pers_hist_loop_norm_mha_dec_list).to(device)

    layer_indx = 0
    mlp_cond2_pers_hist_loop_norm_mha_dec_cat = torch.nn.Sequential()
    layer_indx += 1
    mlp_cond2_pers_hist_loop_norm_mha_dec_cat.add_module(f'mlp{i}_{layer_indx}_linear', torch.nn.Linear(16*mlp_cond2_pers_hist_loop_norm_n_heads_att, len(train_dataset.hist_buckets_arr)-1))
    mlp_cond2_pers_hist_loop_norm_mha_dec_cat = mlp_cond2_pers_hist_loop_norm_mha_dec_cat.to(device)


    #############################

    layer_indx = 0
    mlp_token_cp_pers_cc = torch.nn.Sequential()
    layer_indx += 1
    mlp_token_cp_pers_cc.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, num_clusters))
    mlp_token_cp_pers_cc = mlp_token_cp_pers_cc.to(device)

    layer_indx = 0
    mlp_token_cp_pers_cc_norm = torch.nn.Sequential()
    layer_indx += 1
    mlp_token_cp_pers_cc_norm.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, num_clusters))
    mlp_token_cp_pers_cc_norm = mlp_token_cp_pers_cc_norm.to(device)

    layer_indx = 0
    mlp_token_cp_pers_cc_binary = torch.nn.Sequential()
    layer_indx += 1
    mlp_token_cp_pers_cc_binary.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, num_clusters))
    mlp_token_cp_pers_cc_binary = mlp_token_cp_pers_cc_binary.to(device)

    layer_indx = 0
    mlp_token_cp_pers_loop = torch.nn.Sequential()
    layer_indx += 1
    mlp_token_cp_pers_loop.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, num_clusters))
    mlp_token_cp_pers_loop = mlp_token_cp_pers_loop.to(device)

    layer_indx = 0
    mlp_token_cp_pers_loop_norm = torch.nn.Sequential()
    layer_indx += 1
    mlp_token_cp_pers_loop_norm.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, num_clusters))
    mlp_token_cp_pers_loop_norm = mlp_token_cp_pers_loop_norm.to(device)

    layer_indx = 0
    mlp_token_cp_pers_loop_binary = torch.nn.Sequential()
    layer_indx += 1
    mlp_token_cp_pers_loop_binary.add_module(f'mlp_{layer_indx}_linear', torch.nn.Linear(slide_embedding_size, num_clusters))
    mlp_token_cp_pers_loop_binary = mlp_token_cp_pers_loop_binary.to(device)



    #############################

    if(model_param_path is None):
        init_weights(mlp_project_wsi_embed1)
        init_weights(mlp_project_patch_embed1)

        init_weights(mlp_cond2_prop_clusters_mha_enc)
        init_weights(mlp_cond2_prop_clusters_mha)
        init_weights(mlp_cond2_prop_clusters_wsi_mval)
        init_weights(mlp_cond2_prop_clusters_mha_dec)

        init_weights(mlp_cond2_pers_hist_cc_norm_mha_enc)
        init_weights(mlp_cond2_pers_hist_cc_norm_mha)
        init_weights(mlp_cond2_pers_hist_cc_norm_wsi_mval)
        init_weights(mlp_cond2_pers_hist_cc_norm_mha_dec)
        init_weights(mlp_cond2_pers_hist_cc_norm_mha_dec_cat)

        init_weights(mlp_cond2_pers_hist_loop_norm_mha_enc)
        init_weights(mlp_cond2_pers_hist_loop_norm_mha)
        init_weights(mlp_cond2_pers_hist_loop_norm_wsi_mval)
        init_weights(mlp_cond2_pers_hist_loop_norm_mha_dec)
        init_weights(mlp_cond2_pers_hist_loop_norm_mha_dec_cat)

        init_weights(mlp_token_cp_pers_cc)
        init_weights(mlp_token_cp_pers_cc_norm)
        init_weights(mlp_token_cp_pers_cc_binary)
        init_weights(mlp_token_cp_pers_loop)
        init_weights(mlp_token_cp_pers_loop_norm)
        init_weights(mlp_token_cp_pers_loop_binary)
    else:
        print('loading from ', model_param_path)
        try:
            mlp_project_wsi_embed1.load_state_dict(state_dict2['mlp_project_wsi_embed1'], strict=True)
        except:
            pass
        try:
            mlp_project_patch_embed1.load_state_dict(state_dict2['mlp_project_patch_embed1'], strict=True)
        except:
            pass

        try:
            mlp_cond2_prop_clusters_mha_enc.load_state_dict(state_dict2['mlp_cond2_prop_clusters_mha_enc'], strict=True)
        except:
            pass
        try:
            mlp_cond2_prop_clusters_mha.load_state_dict(state_dict2['mlp_cond2_prop_clusters_mha'], strict=True)
        except:
            pass
        try:
            mlp_cond2_prop_clusters_wsi_mval.load_state_dict(state_dict2['mlp_cond2_prop_clusters_wsi_mval'], strict=True)
        except:
            pass
        try:
            mlp_cond2_prop_clusters_mha_dec.load_state_dict(state_dict2['mlp_cond2_prop_clusters_mha_dec'], strict=True)
        except:
            pass


        try:
            mlp_cond2_pers_hist_cc_norm_mha_enc.load_state_dict(state_dict2['mlp_cond2_pers_hist_cc_norm_mha_enc'], strict=True)
        except:
            pass
        try:
            mlp_cond2_pers_hist_cc_norm_mha.load_state_dict(state_dict2['mlp_cond2_pers_hist_cc_norm_mha'], strict=True)
        except:
            pass
        try:
            mlp_cond2_pers_hist_cc_norm_wsi_mval.load_state_dict(state_dict2['mlp_cond2_pers_hist_cc_norm_wsi_mval'], strict=True)
        except:
            pass
        try:
            mlp_cond2_pers_hist_cc_norm_mha_dec.load_state_dict(state_dict2['mlp_cond2_pers_hist_cc_norm_mha_dec'], strict=True)
        except:
            pass
        try:
            mlp_cond2_pers_hist_cc_norm_mha_dec_cat.load_state_dict(state_dict2['mlp_cond2_pers_hist_cc_norm_mha_dec_cat'], strict=True)
        except:
            pass

        try:
            mlp_cond2_pers_hist_loop_norm_mha_enc.load_state_dict(state_dict2['mlp_cond2_pers_hist_loop_norm_mha_enc'], strict=True)
        except:
            pass
        try:
            mlp_cond2_pers_hist_loop_norm_mha.load_state_dict(state_dict2['mlp_cond2_pers_hist_loop_norm_mha'], strict=True)
        except:
            pass
        try:
            mlp_cond2_pers_hist_loop_norm_wsi_mval.load_state_dict(state_dict2['mlp_cond2_pers_hist_loop_norm_wsi_mval'], strict=True)
        except:
            pass
        try:
            mlp_cond2_pers_hist_loop_norm_mha_dec.load_state_dict(state_dict2['mlp_cond2_pers_hist_loop_norm_mha_dec'], strict=True)
        except:
            pass
        try:
            mlp_cond2_pers_hist_loop_norm_mha_dec_cat.load_state_dict(state_dict2['mlp_cond2_pers_hist_loop_norm_mha_dec_cat'], strict=True)
        except:
            pass

        try:
            mlp_token_cp_pers_cc.load_state_dict(state_dict2['mlp_token_cp_pers_cc'], strict=True)
        except:
            pass
        try:
            mlp_token_cp_pers_cc_norm.load_state_dict(state_dict2['mlp_token_cp_pers_cc_norm'], strict=True)
        except:
            pass
        try:
            mlp_token_cp_pers_cc_binary.load_state_dict(state_dict2['mlp_token_cp_pers_cc_binary'], strict=True)
        except:
            pass
        try:
            mlp_token_cp_pers_loop.load_state_dict(state_dict2['mlp_token_cp_pers_loop'], strict=True)
        except:
            pass
        try:
            mlp_token_cp_pers_loop_norm.load_state_dict(state_dict2['mlp_token_cp_pers_loop_norm'], strict=True)
        except:
            pass
        try:
            mlp_token_cp_pers_loop_binary.load_state_dict(state_dict2['mlp_token_cp_pers_loop_binary'], strict=True)
        except:
            pass
   
        
        
        print('loading done')

    


    # load trainable parameters
    named_parameters = list(model.named_parameters())
    exclude = ( ############ 
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )
    include = lambda n, p: not exclude(n, p)
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    # set optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW([{"params": gain_or_bias_params, "weight_decay": 0.0}, {"params": rest_params, "weight_decay": args.weight_decay}], lr=args.lr)
    optimizer_aux = torch.optim.AdamW([{"params": list(mlp_project_wsi_embed1.parameters()) +list(mlp_project_patch_embed1.parameters())
                                    +list(mlp_cond2_prop_clusters_mha_enc.parameters()) +list(mlp_cond2_prop_clusters_mha.parameters())  +list(mlp_cond2_prop_clusters_wsi_mval.parameters()) +list(mlp_cond2_prop_clusters_mha_dec.parameters())
                                    +list(mlp_cond2_pers_hist_cc_norm_mha_enc.parameters()) +list(mlp_cond2_pers_hist_cc_norm_mha.parameters())  +list(mlp_cond2_pers_hist_cc_norm_wsi_mval.parameters()) +list(mlp_cond2_pers_hist_cc_norm_mha_dec.parameters())+list(mlp_cond2_pers_hist_cc_norm_mha_dec_cat.parameters())
                                    +list(mlp_cond2_pers_hist_loop_norm_mha_enc.parameters()) +list(mlp_cond2_pers_hist_loop_norm_mha.parameters())  +list(mlp_cond2_pers_hist_loop_norm_wsi_mval.parameters()) +list(mlp_cond2_pers_hist_loop_norm_mha_dec.parameters())+list(mlp_cond2_pers_hist_loop_norm_mha_dec_cat.parameters())
                                    +list(mlp_token_cp_pers_cc.parameters()) +list(mlp_token_cp_pers_cc_norm.parameters())  +list(mlp_token_cp_pers_cc_binary.parameters()) +list(mlp_token_cp_pers_loop.parameters())+list(mlp_token_cp_pers_loop_norm.parameters())+list(mlp_token_cp_pers_loop_binary.parameters()),
                                    "weight_decay": args.weight_decay}], lr=args.lr_warmup)

    if(model_param_path is not None):
        try:
            optimizer.load_state_dict(state_dict2['optimizer'], strict=True)
        except:
            pass
        try:
            optimizer_aux.load_state_dict(state_dict2['optimizer_aux'], strict=True)
        except:
            pass

    if(warmup_epochs > 0 and start_epoch <warmup_epochs):
        for param in model.parameters():
            param.requires_grad = False

    lr_scheduler = cosine_lr(
        optimizer=optimizer,
        base_lr=args.lr,
        warmup_length=int(len(train_loader) * args.num_epochs * 0.1),
        steps=(len(train_loader) * args.num_epochs),
    )

    lr_scheduler_aux = cosine_lr(
        optimizer=optimizer_aux,
        base_lr=args.lr,
        warmup_length=int(len(train_loader) * args.num_epochs * 0.1),
        steps=(len(train_loader) * args.num_epochs),
    )


    model.train()
    fp16_scaler = torch.cuda.amp.GradScaler()
    step = 0

    epoch_list = []
    total_loss_list = []

    val_loss_smoothed_arr = np.zeros(args.num_epochs)
    val_loss_smoothed_epoch_arr = np.zeros(args.num_epochs, dtype=int)
    val_loss_smoothed_grad_arr = np.zeros(args.num_epochs)
    early_stop_flag = False

    cond2_prop_clusters_reg_loss_list = []
    cond2_pers_hist_cc_norm_reg_loss_list = []
    cond2_pers_hist_loop_norm_reg_loss_list = []
    cond2_prop_clusters_cumsum_loss_list = []
    cond2_pers_hist_cc_norm_cumsum_loss_list = []
    cond2_pers_hist_loop_norm_cumsum_loss_list = []

    token_cp_pers_cc_loss_list = []
    token_cp_pers_cc_norm_loss_list = []
    token_cp_pers_cc_binary_loss_list = []
    token_cp_pers_loop_loss_list = []
    token_cp_pers_loop_norm_loss_list = []
    token_cp_pers_loop_binary_loss_list = []

    token_similarity_loss_list = []

    #############
    # validation 
    #############

    epoch_val_list = []
    total_loss_val_list = []

    cond2_prop_clusters_reg_loss_val_list = []
    cond2_pers_hist_cc_norm_reg_loss_val_list = []
    cond2_pers_hist_loop_norm_reg_loss_val_list = []
    cond2_prop_clusters_cumsum_loss_val_list = []
    cond2_pers_hist_cc_norm_cumsum_loss_val_list = []
    cond2_pers_hist_loop_norm_cumsum_loss_val_list = []

    token_cp_pers_cc_loss_val_list = []
    token_cp_pers_cc_norm_loss_val_list = []
    token_cp_pers_cc_binary_loss_val_list = []
    token_cp_pers_loop_loss_val_list = []
    token_cp_pers_loop_norm_loss_val_list = []
    token_cp_pers_loop_binary_loss_val_list = []

    token_similarity_loss_val_list = []

    val_e_indx = -1
    for epoch in range(start_epoch, args.num_epochs):
        if(early_stop_flag):
            break
        if(epoch == warmup_epochs):
            for param in model.parameters():
                param.requires_grad = True

        if(epoch >= warmup_epochs):
            model.train()

        mlp_project_wsi_embed1.train()
        mlp_project_patch_embed1.train()

        mlp_cond2_prop_clusters_mha_enc.train()
        mlp_cond2_prop_clusters_mha.train()
        mlp_cond2_prop_clusters_wsi_mval.train()
        mlp_cond2_prop_clusters_mha_dec.train()

        mlp_cond2_pers_hist_cc_norm_mha_enc.train()
        mlp_cond2_pers_hist_cc_norm_mha.train()
        mlp_cond2_pers_hist_cc_norm_wsi_mval.train()
        mlp_cond2_pers_hist_cc_norm_mha_dec.train()
        mlp_cond2_pers_hist_cc_norm_mha_dec_cat.train()

        mlp_cond2_pers_hist_loop_norm_mha_enc.train()
        mlp_cond2_pers_hist_loop_norm_mha.train()
        mlp_cond2_pers_hist_loop_norm_wsi_mval.train()
        mlp_cond2_pers_hist_loop_norm_mha_dec.train()
        mlp_cond2_pers_hist_loop_norm_mha_dec_cat.train()

        mlp_token_cp_pers_cc.train()
        mlp_token_cp_pers_cc_norm.train()
        mlp_token_cp_pers_cc_binary.train()
        mlp_token_cp_pers_loop.train()
        mlp_token_cp_pers_loop_norm.train()
        mlp_token_cp_pers_loop_binary.train()

        epoch_train_loss = 0

        epoch_cond2_prop_clusters_reg_loss = 0
        epoch_cond2_pers_hist_cc_norm_reg_loss = 0
        epoch_cond2_pers_hist_loop_norm_reg_loss = 0
        epoch_cond2_prop_clusters_cumsum_loss = 0
        epoch_cond2_pers_hist_cc_norm_cumsum_loss = 0
        epoch_cond2_pers_hist_loop_norm_cumsum_loss = 0

        epoch_token_cp_pers_cc_loss = 0
        epoch_token_cp_pers_cc_norm_loss = 0
        epoch_token_cp_pers_cc_binary_loss = 0
        epoch_token_cp_pers_loop_loss = 0
        epoch_token_cp_pers_loop_norm_loss = 0
        epoch_token_cp_pers_loop_binary_loss = 0

        epoch_token_similarity_loss = 0


        cond2_prop_clusters_reg_loss = 0
        cond2_pers_hist_cc_norm_reg_loss = 0
        cond2_pers_hist_loop_norm_reg_loss = 0
        cond2_prop_clusters_cumsum_loss = 0
        cond2_pers_hist_cc_norm_cumsum_loss = 0
        cond2_pers_hist_loop_norm_cumsum_loss = 0

        token_cp_pers_cc_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_cc_norm_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_cc_binary_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_loop_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_loop_norm_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_loop_binary_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)

        token_similarity_loss = 0


        token_count = 0
        token_count_bi = 0
        bi = 0
        for indx, (features, coords, patch_size_lv0, cluster_prop_gt, 
                   pers_hist_cc_norm_gt, pers_hist_loop_norm_gt,
                   selected_groups_ids, selected_groups_sample_emb, 
                   token_cp_pers_cc_gt, token_cp_pers_loop_gt, token_cp_pers_cc_norm_gt, token_cp_pers_loop_norm_gt,
                   roi_patch_cluster_ids, roi_patch_embeddings, neg_patch_cluster_ids, neg_patch_embeddings) in enumerate(tqdm(train_loader)):
            bi += 1
            if(epoch>=warmup_epochs):
                lr_scheduler(step)
                lr_scheduler_aux(step)
            features = features.to(device)
            coords = coords.to(device)
            patch_size_lv0 = patch_size_lv0.to(device)
            cluster_prop_gt = cluster_prop_gt.to(device)
            pers_hist_cc_norm_gt = pers_hist_cc_norm_gt.to(device)
            pers_hist_loop_norm_gt = pers_hist_loop_norm_gt.to(device)
            selected_groups_ids = selected_groups_ids.to(device)
            selected_groups_sample_emb = selected_groups_sample_emb.to(device)
            token_cp_pers_cc_gt = token_cp_pers_cc_gt.to(device)
            token_cp_pers_loop_gt = token_cp_pers_loop_gt.to(device)
            token_cp_pers_cc_norm_gt = token_cp_pers_cc_norm_gt.to(device)
            token_cp_pers_loop_norm_gt = token_cp_pers_loop_norm_gt.to(device)

            unique_cluster_ids_src = torch.unique(roi_patch_cluster_ids).detach().cpu().numpy()
            patches_pos_list = []
            patches_neg_list = []
            for cluster_id in unique_cluster_ids_src:
                pos_patches = roi_patch_embeddings[roi_patch_cluster_ids==cluster_id]
                neg_patches = neg_patch_embeddings[neg_patch_cluster_ids==cluster_id]
                if(len(pos_patches.shape)>1 and pos_patches.shape[0]>1 and len(neg_patches.shape)>1 and neg_patches.shape[0]>1 ):
                    count = min(pos_patches.shape[0], neg_patches.shape[0])
                    perm1 = np.random.permutation(np.arange(pos_patches.shape[0]))[:count]
                    perm2 = np.random.permutation(np.arange(neg_patches.shape[0]))[:count]
                    patches_pos_list.append(pos_patches[perm1].to(device))
                    patches_neg_list.append(neg_patches[perm2].to(device))

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                token_similarity_loss = 0
                temperature=0.5
                patches_pos_logits_list = []
                patches_neg_logits_list = []
                for i in range(len(patches_pos_list)):
                    pos_patches_logits = model.forward_features_patch_embed(patches_pos_list[i])
                    neg_patches_logits = model.forward_features_patch_embed(patches_neg_list[i])

                    z = torch.cat([pos_patches_logits, neg_patches_logits], dim=0)
                    z = F.normalize(z, dim=1)
                    z_pos = F.normalize(pos_patches_logits, dim=1)

                    neg_similarity = torch.matmul(z, z.T)
                    N = z_pos.shape[0]
                    mask_neg = (~torch.eye(z.shape[0], dtype=bool))
                    mask_neg[:pos_patches_logits.shape[0],:pos_patches_logits.shape[0]] = 0
                    mask_neg[neg_patches_logits.shape[0]:] = 0
                    count_neg_pairs = mask_neg.sum()
                    mask_neg = mask_neg.to(z.device)
                    sim_neg = neg_similarity / temperature
                    exp_sim_neg = torch.exp(sim_neg) * mask_neg

                    pos_similarity = torch.matmul(z_pos, z_pos.T)
                    N = z_pos.shape[0]
                    mask_pos = (torch.triu(torch.ones(pos_similarity.shape, dtype=bool),diagonal=1))
                    mask_pos = mask_pos.to(z.device)
                    sim_pos = pos_similarity / temperature
                    exp_sim_pos = torch.exp(sim_pos) * mask_pos

                    token_similarity_loss += -torch.log(exp_sim_pos.sum() / exp_sim_neg.sum())

                if(len(patches_pos_list)>0):
                    token_similarity_loss = token_similarity_loss/len(patches_pos_list)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, tokens_logits = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
                print('features train', features.shape)
                print('tokens_logits', tokens_logits.shape)
                print('logits', logits.shape)
                tokens_logits_strict = tokens_logits[:,1:]
                tokens_logits_cls = tokens_logits[:,1]
                # print('tokens_logits_strict', tokens_logits_strict.shape)
                # print('token_cp_pers_cc_gt', token_cp_pers_cc_gt.shape)
                # print('token_count', token_count)
                # print('torch.any(features != 0, dim=-1)', torch.any(features != 0, dim=-1).shape)

                wsi_proj = mlp_project_wsi_embed1(logits)
                cond_tile_proj = mlp_project_patch_embed1(selected_groups_sample_emb)
                if(len(cond_tile_proj.shape)>2):
                    cond_tile_proj = cond_tile_proj.squeeze(dim=0)
                if(cond_tile_proj.shape[0]>1):
                    wsi_proj = wsi_proj.repeat(cond_tile_proj.shape[0],1)
                wsi_tile_cat = torch.cat((wsi_proj, cond_tile_proj), dim=1)
                wsi_tile_cat2 = wsi_tile_cat.clone()

                # print('cond2@@@@@@@@@@@@@@@@@@ cluster prop')
                cond2_prop_clusters_cat_enc = mlp_cond2_prop_clusters_mha_enc(wsi_tile_cat)
                cond2_prop_clusters_att_mha = []
                for i in range(mlp_cond2_prop_clusters_n_heads_att):
                    wsi_proj_val = mlp_cond2_prop_clusters_wsi_mval[i](logits).repeat(wsi_tile_cat.shape[0],1)
                    channel_wise_mha = mlp_cond2_prop_clusters_mha[i](cond2_prop_clusters_cat_enc)
                    elementwise_mul = channel_wise_mha * wsi_proj_val
                    mha_mlp = mlp_cond2_prop_clusters_mha_dec[i](elementwise_mul)
                    cond2_prop_clusters_att_mha.append(mha_mlp) 
                cond2_prop_clusters_pred = cond2_prop_clusters_att_mha[0]
                

                # print('cond2@@@@@@@@@@@@@@@@@@ pers_hist cc norm')
                cond2_pers_hist_cc_norm_cat_enc = mlp_cond2_pers_hist_cc_norm_mha_enc(wsi_tile_cat)
                cond2_pers_hist_cc_norm_att_mha = []
                for i in range(mlp_cond2_pers_hist_cc_norm_n_heads_att):
                    wsi_proj_val = mlp_cond2_pers_hist_cc_norm_wsi_mval[i](logits).repeat(wsi_tile_cat.shape[0],1)
                    channel_wise_mha = mlp_cond2_pers_hist_cc_norm_mha[i](cond2_pers_hist_cc_norm_cat_enc)
                    elementwise_mul = channel_wise_mha * wsi_proj_val
                    mha_mlp = mlp_cond2_pers_hist_cc_norm_mha_dec[i](elementwise_mul)
                    cond2_pers_hist_cc_norm_att_mha.append(mha_mlp) 
                mha_mlp_cat = torch.cat(cond2_pers_hist_cc_norm_att_mha, dim=1)
                cond2_pers_hist_cc_norm_pred = mlp_cond2_pers_hist_cc_norm_mha_dec_cat(mha_mlp_cat)

                # print('cond2@@@@@@@@@@@@@@@@@@ pers_hist loop norm')
                cond2_pers_hist_loop_norm_cat_enc = mlp_cond2_pers_hist_loop_norm_mha_enc(wsi_tile_cat)
                cond2_pers_hist_loop_norm_att_mha = []
                for i in range(mlp_cond2_pers_hist_loop_norm_n_heads_att):
                    wsi_proj_val = mlp_cond2_pers_hist_loop_norm_wsi_mval[i](logits).repeat(wsi_tile_cat.shape[0],1)
                    channel_wise_mha = mlp_cond2_pers_hist_loop_norm_mha[i](cond2_pers_hist_loop_norm_cat_enc)
                    elementwise_mul = channel_wise_mha * wsi_proj_val
                    mha_mlp = mlp_cond2_pers_hist_loop_norm_mha_dec[i](elementwise_mul)
                    cond2_pers_hist_loop_norm_att_mha.append(mha_mlp) 
                mha_mlp_cat = torch.cat(cond2_pers_hist_loop_norm_att_mha, dim=1)
                cond2_pers_hist_loop_norm_pred = mlp_cond2_pers_hist_loop_norm_mha_dec_cat(mha_mlp_cat)

                token_cp_pers_cc_pred = mlp_token_cp_pers_cc(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_cc_norm_pred = mlp_token_cp_pers_cc_norm(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_cc_binary_pred = mlp_token_cp_pers_cc_binary(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_loop_pred = mlp_token_cp_pers_loop(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_loop_norm_pred = mlp_token_cp_pers_loop_norm(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_loop_binary_pred = mlp_token_cp_pers_loop_binary(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))


            cond2_prop_clusters_reg_loss += F.huber_loss(cond2_prop_clusters_pred, cluster_prop_gt[0,selected_groups_ids].squeeze(0).unsqueeze(1),reduction='mean')
            cond2_prop_clusters_cumsum_loss += F.huber_loss(torch.cumsum(cond2_prop_clusters_pred, dim=-1), torch.cumsum(cluster_prop_gt[0,selected_groups_ids].squeeze(0).unsqueeze(1), dim=-1),reduction='mean')

            cond2_pers_hist_cc_norm_reg_loss += F.huber_loss(cond2_pers_hist_cc_norm_pred, pers_hist_cc_norm_gt[0,selected_groups_ids].squeeze(0), reduction='mean')
            cond2_pers_hist_loop_norm_reg_loss += F.huber_loss(cond2_pers_hist_loop_norm_pred, pers_hist_loop_norm_gt[0,selected_groups_ids].squeeze(0), reduction='mean')

            cond2_pers_hist_cc_norm_cumsum_loss += F.huber_loss(torch.cumsum(cond2_pers_hist_cc_norm_pred, dim=-1), torch.cumsum(pers_hist_cc_norm_gt[0,selected_groups_ids].squeeze(0), dim=-1),reduction='mean')
            cond2_pers_hist_loop_norm_cumsum_loss += F.huber_loss(torch.cumsum(cond2_pers_hist_loop_norm_pred, dim=-1), torch.cumsum(pers_hist_loop_norm_gt[0,selected_groups_ids].squeeze(0), dim=-1),reduction='mean')

            if(token_cp_pers_cc_pred.shape[0] == token_cp_pers_cc_gt.squeeze(0).shape[0]):
                token_count += 1
                token_count_bi += 1
                max_values_cc, _ = torch.max(token_cp_pers_cc_gt.squeeze(0),dim=-1)
                max_values_holes, _ = torch.max(token_cp_pers_loop_gt.squeeze(0),dim=-1)

                token_cp_pers_cc_loss_1 = 0.5*(F.huber_loss(token_cp_pers_cc_pred[max_values_cc>0], token_cp_pers_cc_gt.squeeze(0)[max_values_cc>0], reduction='mean'))
                token_cp_pers_cc_norm_loss_1 = 0.5*(F.huber_loss(token_cp_pers_cc_norm_pred[max_values_cc>0], token_cp_pers_cc_norm_gt.squeeze(0)[max_values_cc>0], reduction='mean'))
                token_cp_pers_cc_binary_loss_1 = 0.5*(bce_loss_fn(token_cp_pers_cc_binary_pred[max_values_cc>0], (token_cp_pers_cc_gt>0).squeeze(0)[max_values_cc>0].bfloat16()))

                token_cp_pers_cc_loss_0 = 0.5*(F.huber_loss(token_cp_pers_cc_pred[max_values_cc<=0], token_cp_pers_cc_gt.squeeze(0)[max_values_cc<=0], reduction='mean'))
                token_cp_pers_cc_norm_loss_0 = 0.5*(F.huber_loss(token_cp_pers_cc_norm_pred[max_values_cc<=0], token_cp_pers_cc_norm_gt.squeeze(0)[max_values_cc<=0], reduction='mean'))
                token_cp_pers_cc_binary_loss_0 = 0.5*(bce_loss_fn(token_cp_pers_cc_binary_pred[max_values_cc<=0], (token_cp_pers_cc_gt>0).squeeze(0)[max_values_cc<=0].bfloat16()))

                token_cp_pers_loop_loss_0 = 0.5*(F.huber_loss(token_cp_pers_loop_pred[max_values_holes<=0], token_cp_pers_loop_gt.squeeze(0)[max_values_holes<=0], reduction='mean'))
                token_cp_pers_loop_norm_loss_0 = 0.5*(F.huber_loss(token_cp_pers_loop_norm_pred[max_values_holes<=0], token_cp_pers_loop_norm_gt.squeeze(0)[max_values_holes<=0], reduction='mean'))
                token_cp_pers_loop_binary_loss_0 = 0.5*(bce_loss_fn(token_cp_pers_loop_binary_pred[max_values_holes<=0], (token_cp_pers_loop_gt>0).squeeze(0)[max_values_holes<=0].bfloat16()))

                token_cp_pers_loop_loss_1 = 0.5*(F.huber_loss(token_cp_pers_loop_pred[max_values_holes>0], token_cp_pers_loop_gt.squeeze(0)[max_values_holes>0], reduction='mean'))
                token_cp_pers_loop_norm_loss_1 = 0.5*(F.huber_loss(token_cp_pers_loop_norm_pred[max_values_holes>0], token_cp_pers_loop_norm_gt.squeeze(0)[max_values_holes>0], reduction='mean'))
                token_cp_pers_loop_binary_loss_1 = 0.5*(bce_loss_fn(token_cp_pers_loop_binary_pred[max_values_holes>0], (token_cp_pers_loop_gt>0).squeeze(0)[max_values_holes>0].bfloat16()))

                if(not torch.isnan(token_cp_pers_cc_loss_1)):
                    token_cp_pers_cc_loss += token_cp_pers_cc_loss_1
                # else:
                #     print('token_cp_pers_cc_loss_1 is nan')
                #     print('max_values_cc', max_values_cc)
                if(not torch.isnan(token_cp_pers_cc_loss_0)):
                    token_cp_pers_cc_loss += token_cp_pers_cc_loss_0
                # else:
                #     print('token_cp_pers_cc_loss_0 is nan')
                #     print('max_values_cc', max_values_cc)

                if(not torch.isnan(token_cp_pers_cc_norm_loss_1)):
                    token_cp_pers_cc_norm_loss += token_cp_pers_cc_norm_loss_1
                # else:
                #     print('token_cp_pers_cc_norm_loss_1 is nan')
                #     print('max_values_cc', max_values_cc)
                if(not torch.isnan(token_cp_pers_cc_norm_loss_0)):
                    token_cp_pers_cc_norm_loss += token_cp_pers_cc_norm_loss_0
                # else:
                #     print('token_cp_pers_cc_norm_loss_0 is nan')
                #     print('max_values_cc', max_values_cc)

                if(not torch.isnan(token_cp_pers_cc_binary_loss_1)):
                    token_cp_pers_cc_binary_loss += token_cp_pers_cc_binary_loss_1
                # else:
                #     print('token_cp_pers_cc_binary_loss_1 is nan')
                #     print('max_values_cc', max_values_cc)
                if(not torch.isnan(token_cp_pers_cc_binary_loss_0)):
                    token_cp_pers_cc_binary_loss += token_cp_pers_cc_binary_loss_0
                # else:
                #     print('token_cp_pers_cc_binary_loss_0 is nan')
                #     print('max_values_cc', max_values_cc)


                if(not torch.isnan(token_cp_pers_loop_loss_1)):
                    token_cp_pers_loop_loss += token_cp_pers_loop_loss_1
                # else:
                #     print('token_cp_pers_loop_loss_1 is nan')
                #     print('max_values_holes', max_values_holes)
                if(not torch.isnan(token_cp_pers_loop_loss_0)):
                    token_cp_pers_loop_loss += token_cp_pers_loop_loss_0
                # else:
                #     print('token_cp_pers_loop_loss_0 is nan')
                #     print('max_values_holes', max_values_holes)

                if(not torch.isnan(token_cp_pers_loop_norm_loss_1)):
                    token_cp_pers_loop_norm_loss += token_cp_pers_loop_norm_loss_1
                # else:
                #     print('token_cp_pers_loop_norm_loss_1 is nan')
                #     print('max_values_holes', max_values_holes)
                if(not torch.isnan(token_cp_pers_loop_norm_loss_0)):
                    token_cp_pers_loop_norm_loss += token_cp_pers_loop_norm_loss_0
                # else:
                #     print('token_cp_pers_loop_norm_loss_0 is nan')
                #     print('max_values_holes', max_values_holes)

                if(not torch.isnan(token_cp_pers_loop_binary_loss_1)):
                    token_cp_pers_loop_binary_loss += token_cp_pers_loop_binary_loss_1
                # else:
                #     print('token_cp_pers_loop_binary_loss_1 is nan')
                #     print('max_values_holes', max_values_holes)
                if(not torch.isnan(token_cp_pers_loop_binary_loss_0)):
                    token_cp_pers_loop_binary_loss += token_cp_pers_loop_binary_loss_0
                # else:
                #     print('token_cp_pers_loop_binary_loss_0 is nan')
                #     print('max_values_holes', max_values_holes)

                # token_cp_pers_cc_loss += 0.5*(F.huber_loss(token_cp_pers_cc_pred[max_values_cc>0], token_cp_pers_cc_gt.squeeze(0)[max_values_cc>0], reduction='mean'))
                # token_cp_pers_cc_norm_loss += 0.5*(F.huber_loss(token_cp_pers_cc_norm_pred[max_values_cc>0], token_cp_pers_cc_norm_gt.squeeze(0)[max_values_cc>0], reduction='mean'))
                # token_cp_pers_cc_binary_loss += 0.5*(bce_loss_fn(token_cp_pers_cc_binary_pred[max_values_cc>0], (token_cp_pers_cc_gt>0).squeeze(0)[max_values_cc>0].bfloat16()))

                # token_cp_pers_cc_loss += 0.5*(F.huber_loss(token_cp_pers_cc_pred[max_values_cc<=0], token_cp_pers_cc_gt.squeeze(0)[max_values_cc<=0], reduction='mean'))
                # token_cp_pers_cc_norm_loss += 0.5*(F.huber_loss(token_cp_pers_cc_norm_pred[max_values_cc<=0], token_cp_pers_cc_norm_gt.squeeze(0)[max_values_cc<=0], reduction='mean'))
                # token_cp_pers_cc_binary_loss += 0.5*(bce_loss_fn(token_cp_pers_cc_binary_pred[max_values_cc<=0], (token_cp_pers_cc_gt>0).squeeze(0)[max_values_cc<=0].bfloat16()))

                # token_cp_pers_loop_loss += 0.5*(F.huber_loss(token_cp_pers_loop_pred[max_values_holes<=0], token_cp_pers_loop_gt.squeeze(0)[max_values_holes<=0], reduction='mean'))
                # token_cp_pers_loop_norm_loss += 0.5*(F.huber_loss(token_cp_pers_loop_norm_pred[max_values_holes<=0], token_cp_pers_loop_norm_gt.squeeze(0)[max_values_holes<=0], reduction='mean'))
                # token_cp_pers_loop_binary_loss += 0.5*(bce_loss_fn(token_cp_pers_loop_binary_pred[max_values_holes<=0], (token_cp_pers_loop_gt>0).squeeze(0)[max_values_holes<=0].bfloat16()))

                # token_cp_pers_loop_loss += 0.5*(F.huber_loss(token_cp_pers_loop_pred[max_values_holes>0], token_cp_pers_loop_gt.squeeze(0)[max_values_holes>0], reduction='mean'))
                # token_cp_pers_loop_norm_loss += 0.5*(F.huber_loss(token_cp_pers_loop_norm_pred[max_values_holes>0], token_cp_pers_loop_norm_gt.squeeze(0)[max_values_holes>0], reduction='mean'))
                # token_cp_pers_loop_binary_loss += 0.5*(bce_loss_fn(token_cp_pers_loop_binary_pred[max_values_holes>0], (token_cp_pers_loop_gt>0).squeeze(0)[max_values_holes>0].bfloat16()))

            
            if(bi == args.batch_size or indx==len(train_loader)-1):
                # print('token_count_bi', token_count_bi)
                if(token_count_bi > 0):
                    epoch_token_cp_pers_cc_loss += token_cp_pers_cc_loss.item()
                    epoch_token_cp_pers_cc_norm_loss += token_cp_pers_cc_norm_loss.item()
                    epoch_token_cp_pers_cc_binary_loss += token_cp_pers_cc_binary_loss.item()
                    epoch_token_cp_pers_loop_loss += token_cp_pers_loop_loss.item()
                    epoch_token_cp_pers_loop_norm_loss += token_cp_pers_loop_norm_loss.item()
                    epoch_token_cp_pers_loop_binary_loss += token_cp_pers_loop_binary_loss.item()

                    # print('epoch_token_cp_pers_cc_loss', epoch_token_cp_pers_cc_loss)
                    # print('epoch_token_cp_pers_cc_norm_loss', epoch_token_cp_pers_cc_norm_loss)
                    # print('epoch_token_cp_pers_cc_binary_loss', epoch_token_cp_pers_cc_binary_loss)
                    # print('epoch_token_cp_pers_loop_loss', epoch_token_cp_pers_loop_loss)
                    # print('epoch_token_cp_pers_loop_norm_loss', epoch_token_cp_pers_loop_norm_loss)
                    # print('epoch_token_cp_pers_loop_binary_loss', epoch_token_cp_pers_loop_binary_loss)

                if(token_count_bi == 0): # prevent division by zero
                    token_count_bi = 1
                ##################
                batch_train_loss = (cond2_pers_hist_cc_norm_reg_loss/bi + cond2_pers_hist_loop_norm_reg_loss/bi) * cond2_pers_hist_norm_loss_weight \
                                  + (cond2_prop_clusters_reg_loss/bi) * cond2_cluster_prop_loss_weight \
                                  + (cond2_pers_hist_cc_norm_cumsum_loss/bi + cond2_pers_hist_loop_norm_cumsum_loss/bi) * cond2_pers_hist_norm_cumsum_loss_weight \
                                  + (cond2_prop_clusters_cumsum_loss/bi) * cond2_cluster_prop_cumsum_loss_weight \
                                  + (token_cp_pers_cc_loss/token_count_bi + token_cp_pers_loop_loss/token_count_bi) * token_cp_pers_loss_weight \
                                  + (token_cp_pers_cc_norm_loss/token_count_bi + token_cp_pers_loop_norm_loss/token_count_bi) * token_cp_pers_norm_loss_weight \
                                  + (token_cp_pers_cc_binary_loss/token_count_bi + token_cp_pers_loop_binary_loss/token_count_bi) * token_cp_pers_binary_loss_weight \
                                  + token_similarity_loss * token_similarity_loss_weight

                fp16_scaler.scale(batch_train_loss).backward()
                fp16_scaler.unscale_(optimizer)
                fp16_scaler.unscale_(optimizer_aux)
                fp16_scaler.step(optimizer_aux)
                if(epoch>=warmup_epochs):
                    fp16_scaler.step(optimizer)
                    step += 1
                fp16_scaler.update()
                optimizer.zero_grad()
                optimizer_aux.zero_grad()


                bi = 0
                token_count_bi = 0

                epoch_train_loss += batch_train_loss.item()/len(train_dataset)
                epoch_cond2_pers_hist_cc_norm_reg_loss += cond2_pers_hist_cc_norm_reg_loss.item()/len(train_dataset)
                epoch_cond2_pers_hist_loop_norm_reg_loss += cond2_pers_hist_loop_norm_reg_loss.item()/len(train_dataset)
                epoch_cond2_prop_clusters_reg_loss += cond2_prop_clusters_reg_loss.item()/len(train_dataset)

                epoch_cond2_pers_hist_cc_norm_cumsum_loss += cond2_pers_hist_cc_norm_cumsum_loss.item()/len(train_dataset)
                epoch_cond2_pers_hist_loop_norm_cumsum_loss += cond2_pers_hist_loop_norm_cumsum_loss.item()/len(train_dataset)
                epoch_cond2_prop_clusters_cumsum_loss += cond2_prop_clusters_cumsum_loss.item()/len(train_dataset)



                try:
                    epoch_token_similarity_loss += token_similarity_loss.item()/len(train_dataset)
                except:
                    pass

                step += 1

                cond2_prop_clusters_reg_loss = 0
                cond2_pers_hist_cc_norm_reg_loss = 0
                cond2_pers_hist_loop_norm_reg_loss = 0

                cond2_prop_clusters_cumsum_loss = 0
                cond2_pers_hist_cc_norm_cumsum_loss = 0
                cond2_pers_hist_loop_norm_cumsum_loss = 0

                token_cp_pers_cc_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
                token_cp_pers_cc_norm_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
                token_cp_pers_cc_binary_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
                token_cp_pers_loop_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
                token_cp_pers_loop_norm_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
                token_cp_pers_loop_binary_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)

                token_similarity_loss = 0
            

        # print('epoch_token_cp_pers_cc_loss', epoch_token_cp_pers_cc_loss)
        # print('epoch_token_cp_pers_cc_norm_loss', epoch_token_cp_pers_cc_norm_loss)
        # print('epoch_token_cp_pers_cc_binary_loss', epoch_token_cp_pers_cc_binary_loss)
        # print('epoch_token_cp_pers_loop_loss', epoch_token_cp_pers_loop_loss)
        # print('epoch_token_cp_pers_loop_norm_loss', epoch_token_cp_pers_loop_norm_loss)
        # print('epoch_token_cp_pers_loop_binary_loss', epoch_token_cp_pers_loop_binary_loss)
        

        epoch_token_cp_pers_cc_loss /= token_count
        epoch_token_cp_pers_cc_norm_loss /= token_count
        epoch_token_cp_pers_cc_binary_loss /= token_count
        epoch_token_cp_pers_loop_loss /= token_count
        epoch_token_cp_pers_loop_norm_loss /= token_count
        epoch_token_cp_pers_loop_binary_loss /= token_count

        # print('token_count', token_count)
        # print('epoch_token_cp_pers_cc_loss', epoch_token_cp_pers_cc_loss)
        # print('epoch_token_cp_pers_cc_norm_loss', epoch_token_cp_pers_cc_norm_loss)
        # print('epoch_token_cp_pers_cc_binary_loss', epoch_token_cp_pers_cc_binary_loss)
        # print('epoch_token_cp_pers_loop_loss', epoch_token_cp_pers_loop_loss)
        # print('epoch_token_cp_pers_loop_norm_loss', epoch_token_cp_pers_loop_norm_loss)
        # print('epoch_token_cp_pers_loop_binary_loss', epoch_token_cp_pers_loop_binary_loss)

        epoch_list.append(epoch)
        total_loss_list.append(epoch_train_loss)
        if(epoch > warmup_epochs and epoch_train_loss > 50): # divergence
            early_stop_flag = True

        cond2_prop_clusters_reg_loss_list.append(epoch_cond2_prop_clusters_reg_loss)
        cond2_pers_hist_cc_norm_reg_loss_list.append(epoch_cond2_pers_hist_cc_norm_reg_loss)
        cond2_pers_hist_loop_norm_reg_loss_list.append(epoch_cond2_pers_hist_loop_norm_reg_loss)

        cond2_prop_clusters_cumsum_loss_list.append(epoch_cond2_prop_clusters_cumsum_loss)
        cond2_pers_hist_cc_norm_cumsum_loss_list.append(epoch_cond2_pers_hist_cc_norm_cumsum_loss)
        cond2_pers_hist_loop_norm_cumsum_loss_list.append(epoch_cond2_pers_hist_loop_norm_cumsum_loss)

        token_cp_pers_cc_loss_list.append(epoch_token_cp_pers_cc_loss)
        token_cp_pers_cc_norm_loss_list.append(epoch_token_cp_pers_cc_norm_loss)
        token_cp_pers_cc_binary_loss_list.append(epoch_token_cp_pers_cc_binary_loss)
        token_cp_pers_loop_loss_list.append(epoch_token_cp_pers_loop_loss)
        token_cp_pers_loop_norm_loss_list.append(epoch_token_cp_pers_loop_norm_loss)
        token_cp_pers_loop_binary_loss_list.append(epoch_token_cp_pers_loop_binary_loss)

        token_similarity_loss_list.append(epoch_token_similarity_loss)

        losses_df = pd.DataFrame()
        losses_df['epoch'] = epoch_list
        losses_df['total_loss'] = total_loss_list

        losses_df['cond2_prop_clusters_reg_loss'] = cond2_prop_clusters_reg_loss_list
        losses_df['cond2_pers_hist_cc_norm_reg_loss'] = cond2_pers_hist_cc_norm_reg_loss_list
        losses_df['cond2_pers_hist_loop_norm_reg_loss'] = cond2_pers_hist_loop_norm_reg_loss_list
        losses_df['cond2_prop_clusters_cumsum_loss'] = cond2_prop_clusters_cumsum_loss_list
        losses_df['cond2_pers_hist_cc_norm_cumsum_loss'] = cond2_pers_hist_cc_norm_cumsum_loss_list
        losses_df['cond2_pers_hist_loop_norm_cumsum_loss'] = cond2_pers_hist_loop_norm_cumsum_loss_list

        # print('token_cp_pers_cc_loss_list', token_cp_pers_cc_loss_list)
        # print('token_cp_pers_cc_norm_loss_list', token_cp_pers_cc_norm_loss_list)
        # print('token_cp_pers_cc_binary_loss_list', token_cp_pers_cc_binary_loss_list)
        # print('token_cp_pers_loop_loss_list', token_cp_pers_loop_loss_list)
        # print('token_cp_pers_loop_norm_loss_list', token_cp_pers_loop_norm_loss_list)
        # print('token_cp_pers_loop_binary_loss_list', token_cp_pers_loop_binary_loss_list)

        losses_df['token_cp_pers_cc_loss'] = token_cp_pers_cc_loss_list
        losses_df['token_cp_pers_cc_norm_loss'] = token_cp_pers_cc_norm_loss_list
        losses_df['token_cp_pers_cc_binary_loss'] = token_cp_pers_cc_binary_loss_list
        losses_df['token_cp_pers_loop_loss'] = token_cp_pers_loop_loss_list
        losses_df['token_cp_pers_loop_norm_loss'] = token_cp_pers_loop_norm_loss_list
        losses_df['token_cp_pers_loop_binary_loss'] = token_cp_pers_loop_binary_loss_list

        losses_df['token_similarity_loss'] = token_similarity_loss_list
        

        losses_df.to_csv(os.path.join(checkpoints_save_path, f'epoch_losses_e{epoch_list[0]}.csv'))

        ################################################
        # Validation
        ################################################

        epoch_val_loss = 0
        epoch_cond2_prop_clusters_reg_loss = 0
        epoch_cond2_pers_hist_cc_norm_reg_loss = 0
        epoch_cond2_pers_hist_loop_norm_reg_loss = 0

        epoch_cond2_prop_clusters_cumsum_loss = 0
        epoch_cond2_pers_hist_cc_norm_cumsum_loss = 0
        epoch_cond2_pers_hist_loop_norm_cumsum_loss = 0

        epoch_token_cp_pers_cc_loss = 0
        epoch_token_cp_pers_cc_norm_loss = 0
        epoch_token_cp_pers_cc_binary_loss = 0
        epoch_token_cp_pers_loop_loss = 0
        epoch_token_cp_pers_loop_norm_loss = 0
        epoch_token_cp_pers_loop_binary_loss = 0

        epoch_token_similarity_loss = 0

        cond2_prop_clusters_reg_loss = 0
        cond2_pers_hist_cc_norm_reg_loss = 0
        cond2_pers_hist_loop_norm_reg_loss = 0

        cond2_prop_clusters_cumsum_loss = 0
        cond2_pers_hist_cc_norm_cumsum_loss = 0
        cond2_pers_hist_loop_norm_cumsum_loss = 0

        token_cp_pers_cc_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_cc_norm_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_cc_binary_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_loop_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_loop_norm_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)
        token_cp_pers_loop_binary_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)

        token_similarity_loss = torch.tensor(0, dtype=torch.bfloat16).to(device)

        model.eval()

        mlp_project_wsi_embed1.eval()
        mlp_project_patch_embed1.eval()

        mlp_cond2_prop_clusters_mha_enc.eval()
        mlp_cond2_prop_clusters_mha.eval()
        mlp_cond2_prop_clusters_wsi_mval.eval()
        mlp_cond2_prop_clusters_mha_dec.eval()

        mlp_cond2_pers_hist_cc_norm_mha_enc.eval()
        mlp_cond2_pers_hist_cc_norm_mha.eval()
        mlp_cond2_pers_hist_cc_norm_wsi_mval.eval()
        mlp_cond2_pers_hist_cc_norm_mha_dec.eval()
        mlp_cond2_pers_hist_cc_norm_mha_dec_cat.eval()

        mlp_cond2_pers_hist_loop_norm_mha_enc.eval()
        mlp_cond2_pers_hist_loop_norm_mha.eval()
        mlp_cond2_pers_hist_loop_norm_wsi_mval.eval()
        mlp_cond2_pers_hist_loop_norm_mha_dec.eval()
        mlp_cond2_pers_hist_loop_norm_mha_dec_cat.eval()

        mlp_token_cp_pers_cc.eval()
        mlp_token_cp_pers_cc_norm.eval()
        mlp_token_cp_pers_cc_binary.eval()
        mlp_token_cp_pers_loop.eval()
        mlp_token_cp_pers_loop_norm.eval()
        mlp_token_cp_pers_loop_binary.eval()

        token_count = 0
        token_count_bi = 0
        bi = 0
        
        for indx, (features, coords, patch_size_lv0, cluster_prop_gt, 
                    pers_hist_cc_norm_gt, pers_hist_loop_norm_gt,
                    selected_groups_ids, selected_groups_sample_emb, 
                    token_cp_pers_cc_gt, token_cp_pers_loop_gt, token_cp_pers_cc_norm_gt, token_cp_pers_loop_norm_gt,
                    roi_patch_cluster_ids, roi_patch_embeddings, neg_patch_cluster_ids, neg_patch_embeddings) in enumerate(tqdm(val_loader)):
            bi += 1
            features = features.to(device)
            coords = coords.to(device)
            patch_size_lv0 = patch_size_lv0.to(device)
            cluster_prop_gt = cluster_prop_gt.to(device)
            pers_hist_cc_norm_gt = pers_hist_cc_norm_gt.to(device)
            pers_hist_loop_norm_gt = pers_hist_loop_norm_gt.to(device)
            selected_groups_ids = selected_groups_ids.to(device)
            selected_groups_sample_emb = selected_groups_sample_emb.to(device)
            token_cp_pers_cc_gt = token_cp_pers_cc_gt.to(device)
            token_cp_pers_loop_gt = token_cp_pers_loop_gt.to(device)
            token_cp_pers_cc_norm_gt = token_cp_pers_cc_norm_gt.to(device)
            token_cp_pers_loop_norm_gt = token_cp_pers_loop_norm_gt.to(device)

            unique_cluster_ids_src = torch.unique(roi_patch_cluster_ids).detach().cpu().numpy()
            patches_pos_list = []
            patches_neg_list = []
            for cluster_id in unique_cluster_ids_src:
                pos_patches = roi_patch_embeddings[roi_patch_cluster_ids==cluster_id]
                neg_patches = neg_patch_embeddings[neg_patch_cluster_ids==cluster_id]
                if(len(pos_patches.shape)>1 and pos_patches.shape[0]>1 and len(neg_patches.shape)>1 and neg_patches.shape[0]>1 ):
                    count = min(pos_patches.shape[0], neg_patches.shape[0])
                    perm1 = np.random.permutation(np.arange(pos_patches.shape[0]))[:count]
                    perm2 = np.random.permutation(np.arange(neg_patches.shape[0]))[:count]
                    patches_pos_list.append(pos_patches[perm1].to(device))
                    patches_neg_list.append(neg_patches[perm2].to(device))

            with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
                token_similarity_loss = 0
                temperature=0.5
                patches_pos_logits_list = []
                patches_neg_logits_list = []
                for i in range(len(patches_pos_list)):
                    pos_patches_logits = model.forward_features_patch_embed(patches_pos_list[i])
                    neg_patches_logits = model.forward_features_patch_embed(patches_neg_list[i])

                    z = torch.cat([pos_patches_logits, neg_patches_logits], dim=0)
                    z = F.normalize(z, dim=1)
                    z_pos = F.normalize(pos_patches_logits, dim=1)

                    neg_similarity = torch.matmul(z, z.T)
                    N = z_pos.shape[0]
                    mask_neg = (~torch.eye(z.shape[0], dtype=bool))
                    mask_neg[:pos_patches_logits.shape[0],:pos_patches_logits.shape[0]] = 0
                    mask_neg[neg_patches_logits.shape[0]:] = 0
                    count_neg_pairs = mask_neg.sum()
                    mask_neg = mask_neg.to(z.device)
                    sim_neg = neg_similarity / temperature
                    exp_sim_neg = torch.exp(sim_neg) * mask_neg

                    pos_similarity = torch.matmul(z_pos, z_pos.T)
                    N = z_pos.shape[0]
                    mask_pos = (torch.triu(torch.ones(pos_similarity.shape, dtype=bool),diagonal=1))
                    mask_pos = mask_pos.to(z.device)
                    sim_pos = pos_similarity / temperature
                    exp_sim_pos = torch.exp(sim_pos) * mask_pos

                    token_similarity_loss += -torch.log(exp_sim_pos.sum() / exp_sim_neg.sum())

                if(len(patches_pos_list)>0):
                    token_similarity_loss = token_similarity_loss/len(patches_pos_list)


            with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
                logits, tokens_logits = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
                tokens_logits_strict = tokens_logits[:,1:]
                tokens_logits_cls = tokens_logits[:,1]
                print('features', features.shape)
                print('tokens_logits_strict', tokens_logits_strict.shape)
                print('token_cp_pers_cc_gt', token_cp_pers_cc_gt.shape)
                print('token_count', token_count)

                wsi_proj = mlp_project_wsi_embed1(logits)
                cond_tile_proj = mlp_project_patch_embed1(selected_groups_sample_emb)
                if(len(cond_tile_proj.shape)>2):
                    cond_tile_proj = cond_tile_proj.squeeze(dim=0)
                if(cond_tile_proj.shape[0]>1):
                    wsi_proj = wsi_proj.repeat(cond_tile_proj.shape[0],1)
                wsi_tile_cat = torch.cat((wsi_proj, cond_tile_proj), dim=1)
                wsi_tile_cat2 = wsi_tile_cat.clone()

                # print('cond2@@@@@@@@@@@@@@@@@@ cluster prop')
                cond2_prop_clusters_cat_enc = mlp_cond2_prop_clusters_mha_enc(wsi_tile_cat)
                cond2_prop_clusters_att_mha = []
                for i in range(mlp_cond2_prop_clusters_n_heads_att):
                    wsi_proj_val = mlp_cond2_prop_clusters_wsi_mval[i](logits).repeat(wsi_tile_cat.shape[0],1)
                    channel_wise_mha = mlp_cond2_prop_clusters_mha[i](cond2_prop_clusters_cat_enc)
                    elementwise_mul = channel_wise_mha * wsi_proj_val
                    mha_mlp = mlp_cond2_prop_clusters_mha_dec[i](elementwise_mul)
                    cond2_prop_clusters_att_mha.append(mha_mlp) 
                cond2_prop_clusters_pred = cond2_prop_clusters_att_mha[0]
                

                # print('cond2@@@@@@@@@@@@@@@@@@ pers_hist cc norm')
                cond2_pers_hist_cc_norm_cat_enc = mlp_cond2_pers_hist_cc_norm_mha_enc(wsi_tile_cat)
                cond2_pers_hist_cc_norm_att_mha = []
                for i in range(mlp_cond2_pers_hist_cc_norm_n_heads_att):
                    wsi_proj_val = mlp_cond2_pers_hist_cc_norm_wsi_mval[i](logits).repeat(wsi_tile_cat.shape[0],1)
                    channel_wise_mha = mlp_cond2_pers_hist_cc_norm_mha[i](cond2_pers_hist_cc_norm_cat_enc)
                    elementwise_mul = channel_wise_mha * wsi_proj_val
                    mha_mlp = mlp_cond2_pers_hist_cc_norm_mha_dec[i](elementwise_mul)
                    cond2_pers_hist_cc_norm_att_mha.append(mha_mlp) 
                mha_mlp_cat = torch.cat(cond2_pers_hist_cc_norm_att_mha, dim=1)
                cond2_pers_hist_cc_norm_pred = mlp_cond2_pers_hist_cc_norm_mha_dec_cat(mha_mlp_cat)

                # print('cond2@@@@@@@@@@@@@@@@@@ pers_hist loop norm')
                cond2_pers_hist_loop_norm_cat_enc = mlp_cond2_pers_hist_loop_norm_mha_enc(wsi_tile_cat)
                cond2_pers_hist_loop_norm_att_mha = []
                for i in range(mlp_cond2_pers_hist_loop_norm_n_heads_att):
                    wsi_proj_val = mlp_cond2_pers_hist_loop_norm_wsi_mval[i](logits).repeat(wsi_tile_cat.shape[0],1)
                    channel_wise_mha = mlp_cond2_pers_hist_loop_norm_mha[i](cond2_pers_hist_loop_norm_cat_enc)
                    elementwise_mul = channel_wise_mha * wsi_proj_val
                    mha_mlp = mlp_cond2_pers_hist_loop_norm_mha_dec[i](elementwise_mul)
                    cond2_pers_hist_loop_norm_att_mha.append(mha_mlp) 
                mha_mlp_cat = torch.cat(cond2_pers_hist_loop_norm_att_mha, dim=1)
                cond2_pers_hist_loop_norm_pred = mlp_cond2_pers_hist_loop_norm_mha_dec_cat(mha_mlp_cat)


                token_cp_pers_cc_pred = mlp_token_cp_pers_cc(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_cc_norm_pred = mlp_token_cp_pers_cc_norm(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_cc_binary_pred = mlp_token_cp_pers_cc_binary(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_loop_pred = mlp_token_cp_pers_loop(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_loop_norm_pred = mlp_token_cp_pers_loop_norm(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))
                token_cp_pers_loop_binary_pred = mlp_token_cp_pers_loop_binary(tokens_logits_strict.view(-1, tokens_logits_strict.shape[-1]))


            cond2_prop_clusters_reg_loss += F.huber_loss(cond2_prop_clusters_pred, cluster_prop_gt[0,selected_groups_ids].squeeze(0).unsqueeze(1),reduction='mean')
            cond2_prop_clusters_cumsum_loss += F.huber_loss(torch.cumsum(cond2_prop_clusters_pred, dim=-1), torch.cumsum(cluster_prop_gt[0,selected_groups_ids].squeeze(0).unsqueeze(1), dim=-1),reduction='mean')

            cond2_pers_hist_cc_norm_reg_loss += F.huber_loss(cond2_pers_hist_cc_norm_pred, pers_hist_cc_norm_gt[0,selected_groups_ids].squeeze(0), reduction='mean')
            cond2_pers_hist_loop_norm_reg_loss += F.huber_loss(cond2_pers_hist_loop_norm_pred, pers_hist_loop_norm_gt[0,selected_groups_ids].squeeze(0), reduction='mean')

            cond2_pers_hist_cc_norm_cumsum_loss += F.huber_loss(torch.cumsum(cond2_pers_hist_cc_norm_pred, dim=-1), torch.cumsum(pers_hist_cc_norm_gt[0,selected_groups_ids].squeeze(0), dim=-1),reduction='mean')
            cond2_pers_hist_loop_norm_cumsum_loss += F.huber_loss(torch.cumsum(cond2_pers_hist_loop_norm_pred, dim=-1), torch.cumsum(pers_hist_loop_norm_gt[0,selected_groups_ids].squeeze(0), dim=-1),reduction='mean')

            if(token_cp_pers_cc_pred.shape[0] == token_cp_pers_cc_gt.squeeze(0).shape[0]):
                token_count += 1
                token_count_bi += 1
                max_values_cc, _ = torch.max(token_cp_pers_cc_gt.squeeze(0),dim=-1)
                max_values_holes, _ = torch.max(token_cp_pers_loop_gt.squeeze(0),dim=-1)

                token_cp_pers_cc_loss_1 = 0.5*(F.huber_loss(token_cp_pers_cc_pred[max_values_cc>0], token_cp_pers_cc_gt.squeeze(0)[max_values_cc>0], reduction='mean'))
                token_cp_pers_cc_norm_loss_1 = 0.5*(F.huber_loss(token_cp_pers_cc_norm_pred[max_values_cc>0], token_cp_pers_cc_norm_gt.squeeze(0)[max_values_cc>0], reduction='mean'))
                token_cp_pers_cc_binary_loss_1 = 0.5*(bce_loss_fn(token_cp_pers_cc_binary_pred[max_values_cc>0], (token_cp_pers_cc_gt>0).squeeze(0)[max_values_cc>0].bfloat16()))

                token_cp_pers_cc_loss_0 = 0.5*(F.huber_loss(token_cp_pers_cc_pred[max_values_cc<=0], token_cp_pers_cc_gt.squeeze(0)[max_values_cc<=0], reduction='mean'))
                token_cp_pers_cc_norm_loss_0 = 0.5*(F.huber_loss(token_cp_pers_cc_norm_pred[max_values_cc<=0], token_cp_pers_cc_norm_gt.squeeze(0)[max_values_cc<=0], reduction='mean'))
                token_cp_pers_cc_binary_loss_0 = 0.5*(bce_loss_fn(token_cp_pers_cc_binary_pred[max_values_cc<=0], (token_cp_pers_cc_gt>0).squeeze(0)[max_values_cc<=0].bfloat16()))

                token_cp_pers_loop_loss_0 = 0.5*(F.huber_loss(token_cp_pers_loop_pred[max_values_holes<=0], token_cp_pers_loop_gt.squeeze(0)[max_values_holes<=0], reduction='mean'))
                token_cp_pers_loop_norm_loss_0 = 0.5*(F.huber_loss(token_cp_pers_loop_norm_pred[max_values_holes<=0], token_cp_pers_loop_norm_gt.squeeze(0)[max_values_holes<=0], reduction='mean'))
                token_cp_pers_loop_binary_loss_0 = 0.5*(bce_loss_fn(token_cp_pers_loop_binary_pred[max_values_holes<=0], (token_cp_pers_loop_gt>0).squeeze(0)[max_values_holes<=0].bfloat16()))

                token_cp_pers_loop_loss_1 = 0.5*(F.huber_loss(token_cp_pers_loop_pred[max_values_holes>0], token_cp_pers_loop_gt.squeeze(0)[max_values_holes>0], reduction='mean'))
                token_cp_pers_loop_norm_loss_1 = 0.5*(F.huber_loss(token_cp_pers_loop_norm_pred[max_values_holes>0], token_cp_pers_loop_norm_gt.squeeze(0)[max_values_holes>0], reduction='mean'))
                token_cp_pers_loop_binary_loss_1 = 0.5*(bce_loss_fn(token_cp_pers_loop_binary_pred[max_values_holes>0], (token_cp_pers_loop_gt>0).squeeze(0)[max_values_holes>0].bfloat16()))

                if(not torch.isnan(token_cp_pers_cc_loss_1)):
                    token_cp_pers_cc_loss += token_cp_pers_cc_loss_1
                # else:
                #     print('token_cp_pers_cc_loss_1 is nan')
                #     print('max_values_cc', max_values_cc)
                if(not torch.isnan(token_cp_pers_cc_loss_0)):
                    token_cp_pers_cc_loss += token_cp_pers_cc_loss_0
                # else:
                #     print('token_cp_pers_cc_loss_0 is nan')
                #     print('max_values_cc', max_values_cc)

                if(not torch.isnan(token_cp_pers_cc_norm_loss_1)):
                    token_cp_pers_cc_norm_loss += token_cp_pers_cc_norm_loss_1
                # else:
                #     print('token_cp_pers_cc_norm_loss_1 is nan')
                #     print('max_values_cc', max_values_cc)
                if(not torch.isnan(token_cp_pers_cc_norm_loss_0)):
                    token_cp_pers_cc_norm_loss += token_cp_pers_cc_norm_loss_0
                # else:
                #     print('token_cp_pers_cc_norm_loss_0 is nan')
                #     print('max_values_cc', max_values_cc)

                if(not torch.isnan(token_cp_pers_cc_binary_loss_1)):
                    token_cp_pers_cc_binary_loss += token_cp_pers_cc_binary_loss_1
                # else:
                #     print('token_cp_pers_cc_binary_loss_1 is nan')
                #     print('max_values_cc', max_values_cc)
                if(not torch.isnan(token_cp_pers_cc_binary_loss_0)):
                    token_cp_pers_cc_binary_loss += token_cp_pers_cc_binary_loss_0
                # else:
                #     print('token_cp_pers_cc_binary_loss_0 is nan')
                #     print('max_values_cc', max_values_cc)


                if(not torch.isnan(token_cp_pers_loop_loss_1)):
                    token_cp_pers_loop_loss += token_cp_pers_loop_loss_1
                # else:
                #     print('token_cp_pers_loop_loss_1 is nan')
                #     print('max_values_holes', max_values_holes)
                if(not torch.isnan(token_cp_pers_loop_loss_0)):
                    token_cp_pers_loop_loss += token_cp_pers_loop_loss_0
                # else:
                #     print('token_cp_pers_loop_loss_0 is nan')
                #     print('max_values_holes', max_values_holes)

                if(not torch.isnan(token_cp_pers_loop_norm_loss_1)):
                    token_cp_pers_loop_norm_loss += token_cp_pers_loop_norm_loss_1
                # else:
                #     print('token_cp_pers_loop_norm_loss_1 is nan')
                #     print('max_values_holes', max_values_holes)
                if(not torch.isnan(token_cp_pers_loop_norm_loss_0)):
                    token_cp_pers_loop_norm_loss += token_cp_pers_loop_norm_loss_0
                # else:
                #     print('token_cp_pers_loop_norm_loss_0 is nan')
                #     print('max_values_holes', max_values_holes)

                if(not torch.isnan(token_cp_pers_loop_binary_loss_1)):
                    token_cp_pers_loop_binary_loss += token_cp_pers_loop_binary_loss_1
                # else:
                #     print('token_cp_pers_loop_binary_loss_1 is nan')
                #     print('max_values_holes', max_values_holes)
                if(not torch.isnan(token_cp_pers_loop_binary_loss_0)):
                    token_cp_pers_loop_binary_loss += token_cp_pers_loop_binary_loss_0
                # else:
                #     print('token_cp_pers_loop_binary_loss_0 is nan')
                #     print('max_values_holes', max_values_holes)

                # token_cp_pers_cc_loss += 0.5*(F.huber_loss(token_cp_pers_cc_pred[max_values_cc>0], token_cp_pers_cc_gt.squeeze(0)[max_values_cc>0], reduction='mean'))
                # token_cp_pers_cc_norm_loss += 0.5*(F.huber_loss(token_cp_pers_cc_norm_pred[max_values_cc>0], token_cp_pers_cc_norm_gt.squeeze(0)[max_values_cc>0], reduction='mean'))
                # token_cp_pers_cc_binary_loss += 0.5*(bce_loss_fn(token_cp_pers_cc_binary_pred[max_values_cc>0], (token_cp_pers_cc_gt>0).squeeze(0)[max_values_cc>0].bfloat16()))

                # token_cp_pers_cc_loss += 0.5*(F.huber_loss(token_cp_pers_cc_pred[max_values_cc<=0], token_cp_pers_cc_gt.squeeze(0)[max_values_cc<=0], reduction='mean'))
                # token_cp_pers_cc_norm_loss += 0.5*(F.huber_loss(token_cp_pers_cc_norm_pred[max_values_cc<=0], token_cp_pers_cc_norm_gt.squeeze(0)[max_values_cc<=0], reduction='mean'))
                # token_cp_pers_cc_binary_loss += 0.5*(bce_loss_fn(token_cp_pers_cc_binary_pred[max_values_cc<=0], (token_cp_pers_cc_gt>0).squeeze(0)[max_values_cc<=0].bfloat16()))

                # token_cp_pers_loop_loss += 0.5*(F.huber_loss(token_cp_pers_loop_pred[max_values_holes<=0], token_cp_pers_loop_gt.squeeze(0)[max_values_holes<=0], reduction='mean'))
                # token_cp_pers_loop_norm_loss += 0.5*(F.huber_loss(token_cp_pers_loop_norm_pred[max_values_holes<=0], token_cp_pers_loop_norm_gt.squeeze(0)[max_values_holes<=0], reduction='mean'))
                # token_cp_pers_loop_binary_loss += 0.5*(bce_loss_fn(token_cp_pers_loop_binary_pred[max_values_holes<=0], (token_cp_pers_loop_gt>0).squeeze(0)[max_values_holes<=0].bfloat16()))

                # token_cp_pers_loop_loss += 0.5*(F.huber_loss(token_cp_pers_loop_pred[max_values_holes>0], token_cp_pers_loop_gt.squeeze(0)[max_values_holes>0], reduction='mean'))
                # token_cp_pers_loop_norm_loss += 0.5*(F.huber_loss(token_cp_pers_loop_norm_pred[max_values_holes>0], token_cp_pers_loop_norm_gt.squeeze(0)[max_values_holes>0], reduction='mean'))
                # token_cp_pers_loop_binary_loss += 0.5*(bce_loss_fn(token_cp_pers_loop_binary_pred[max_values_holes>0], (token_cp_pers_loop_gt>0).squeeze(0)[max_values_holes>0].bfloat16()))


            if(indx==len(val_loader)-1):  
                if(token_count_bi == 0):
                    token_count_bi = 1
                ##################
                batch_val_loss = (cond2_pers_hist_cc_norm_reg_loss/bi + cond2_pers_hist_loop_norm_reg_loss/bi) * cond2_pers_hist_norm_loss_weight \
                                    + (cond2_prop_clusters_reg_loss/bi) * cond2_cluster_prop_loss_weight \
                                    + (cond2_pers_hist_cc_norm_cumsum_loss/bi + cond2_pers_hist_loop_norm_cumsum_loss/bi) * cond2_pers_hist_norm_cumsum_loss_weight \
                                    + (cond2_prop_clusters_cumsum_loss/bi) * cond2_cluster_prop_cumsum_loss_weight \
                                    + (token_cp_pers_cc_loss/token_count_bi + token_cp_pers_loop_loss/token_count_bi) * token_cp_pers_loss_weight \
                                    + (token_cp_pers_cc_norm_loss/token_count_bi + token_cp_pers_loop_norm_loss/token_count_bi) * token_cp_pers_norm_loss_weight \
                                    + (token_cp_pers_cc_binary_loss/token_count_bi + token_cp_pers_loop_binary_loss/token_count_bi) * token_cp_pers_binary_loss_weight \
                                    + token_similarity_loss * token_similarity_loss_weight


                epoch_val_loss += batch_val_loss.item()/len(val_dataset)
                if(epoch > warmup_epochs and epoch_val_loss > 50): # divergence
                    early_stop_flag = True

                epoch_cond2_pers_hist_cc_norm_reg_loss += cond2_pers_hist_cc_norm_reg_loss.item()/len(val_dataset)
                epoch_cond2_pers_hist_loop_norm_reg_loss += cond2_pers_hist_loop_norm_reg_loss.item()/len(val_dataset)
                epoch_cond2_prop_clusters_reg_loss += cond2_prop_clusters_reg_loss.item()/len(val_dataset)

                epoch_cond2_pers_hist_cc_norm_cumsum_loss += cond2_pers_hist_cc_norm_cumsum_loss.item()/len(val_dataset)
                epoch_cond2_pers_hist_loop_norm_cumsum_loss += cond2_pers_hist_loop_norm_cumsum_loss.item()/len(val_dataset)
                epoch_cond2_prop_clusters_cumsum_loss += cond2_prop_clusters_cumsum_loss.item()/len(val_dataset)

                if(token_count > 0):
                    epoch_token_cp_pers_cc_loss += token_cp_pers_cc_loss.item()/token_count
                    epoch_token_cp_pers_cc_norm_loss += token_cp_pers_cc_norm_loss.item()/token_count
                    epoch_token_cp_pers_cc_binary_loss += token_cp_pers_cc_binary_loss.item()/token_count
                    epoch_token_cp_pers_loop_loss += token_cp_pers_loop_loss.item()/token_count
                    epoch_token_cp_pers_loop_norm_loss += token_cp_pers_loop_norm_loss.item()/token_count
                    epoch_token_cp_pers_loop_binary_loss += token_cp_pers_loop_binary_loss.item()/token_count

                try:
                    epoch_token_similarity_loss += token_similarity_loss.item()/len(val_dataset)
                except:
                    pass

                epoch_val_list.append(epoch)
                total_loss_val_list.append(epoch_val_loss)

                cond2_prop_clusters_reg_loss_val_list.append(epoch_cond2_prop_clusters_reg_loss)
                cond2_pers_hist_cc_norm_reg_loss_val_list.append(epoch_cond2_pers_hist_cc_norm_reg_loss)
                cond2_pers_hist_loop_norm_reg_loss_val_list.append(epoch_cond2_pers_hist_loop_norm_reg_loss)

                cond2_prop_clusters_cumsum_loss_val_list.append(epoch_cond2_prop_clusters_cumsum_loss)
                cond2_pers_hist_cc_norm_cumsum_loss_val_list.append(epoch_cond2_pers_hist_cc_norm_cumsum_loss)
                cond2_pers_hist_loop_norm_cumsum_loss_val_list.append(epoch_cond2_pers_hist_loop_norm_cumsum_loss)

                token_cp_pers_cc_loss_val_list.append(epoch_token_cp_pers_cc_loss)
                token_cp_pers_cc_norm_loss_val_list.append(epoch_token_cp_pers_cc_norm_loss)
                token_cp_pers_cc_binary_loss_val_list.append(epoch_token_cp_pers_cc_binary_loss)
                token_cp_pers_loop_loss_val_list.append(epoch_token_cp_pers_loop_loss)
                token_cp_pers_loop_norm_loss_val_list.append(epoch_token_cp_pers_loop_norm_loss)
                token_cp_pers_loop_binary_loss_val_list.append(epoch_token_cp_pers_loop_binary_loss)

                token_similarity_loss_val_list.append(epoch_token_similarity_loss)


                losses_val_df = pd.DataFrame()
                losses_val_df['epoch'] = epoch_val_list
                losses_val_df['total_loss'] = total_loss_val_list

                losses_val_df['cond2_prop_clusters_reg_loss'] = cond2_prop_clusters_reg_loss_val_list
                losses_val_df['cond2_pers_hist_cc_norm_reg_loss'] = cond2_pers_hist_cc_norm_reg_loss_val_list
                losses_val_df['cond2_pers_hist_loop_norm_reg_loss'] = cond2_pers_hist_loop_norm_reg_loss_val_list

                losses_val_df['cond2_prop_clusters_cumsum_loss'] = cond2_prop_clusters_cumsum_loss_val_list
                losses_val_df['cond2_pers_hist_cc_norm_cumsum_loss'] = cond2_pers_hist_cc_norm_cumsum_loss_val_list
                losses_val_df['cond2_pers_hist_loop_norm_cumsum_loss'] = cond2_pers_hist_loop_norm_cumsum_loss_val_list

                losses_val_df['token_cp_pers_cc_loss'] = token_cp_pers_cc_loss_val_list
                losses_val_df['token_cp_pers_cc_norm_loss'] = token_cp_pers_cc_norm_loss_val_list
                losses_val_df['token_cp_pers_cc_binary_loss'] = token_cp_pers_cc_binary_loss_val_list
                losses_val_df['token_cp_pers_loop_loss'] = token_cp_pers_loop_loss_val_list
                losses_val_df['token_cp_pers_loop_norm_loss'] = token_cp_pers_loop_norm_loss_val_list
                losses_val_df['token_cp_pers_loop_binary_loss'] = token_cp_pers_loop_binary_loss_val_list

                losses_val_df['token_similarity_loss'] = token_similarity_loss_val_list
        
                losses_val_df.to_csv(os.path.join(checkpoints_save_path, f'epoch_losses_e{epoch_val_list[0]}_val.csv'))


        state_dict = {'toposlide':model.state_dict(),
                        'mlp_project_wsi_embed1':mlp_project_wsi_embed1.state_dict(),
                        'mlp_project_patch_embed1':mlp_project_patch_embed1.state_dict(),
                          
                        'mlp_cond2_prop_clusters_mha_enc':mlp_cond2_prop_clusters_mha_enc.state_dict(),
                        'mlp_cond2_prop_clusters_mha':mlp_cond2_prop_clusters_mha.state_dict(),
                        'mlp_cond2_prop_clusters_wsi_mval':mlp_cond2_prop_clusters_wsi_mval.state_dict(),
                        'mlp_cond2_prop_clusters_mha_dec':mlp_cond2_prop_clusters_mha_dec.state_dict(),
                          
                        'mlp_cond2_pers_hist_cc_norm_mha_enc':mlp_cond2_pers_hist_cc_norm_mha_enc.state_dict(),
                        'mlp_cond2_pers_hist_cc_norm_mha':mlp_cond2_pers_hist_cc_norm_mha.state_dict(),
                        'mlp_cond2_pers_hist_cc_norm_wsi_mval':mlp_cond2_pers_hist_cc_norm_wsi_mval.state_dict(),
                        'mlp_cond2_pers_hist_cc_norm_mha_dec':mlp_cond2_pers_hist_cc_norm_mha_dec.state_dict(),
                        'mlp_cond2_pers_hist_cc_norm_mha_dec_cat':mlp_cond2_pers_hist_cc_norm_mha_dec_cat.state_dict(),

                        'mlp_cond2_pers_hist_loop_norm_mha_enc':mlp_cond2_pers_hist_loop_norm_mha_enc.state_dict(),
                        'mlp_cond2_pers_hist_loop_norm_mha':mlp_cond2_pers_hist_loop_norm_mha.state_dict(),
                        'mlp_cond2_pers_hist_loop_norm_wsi_mval':mlp_cond2_pers_hist_loop_norm_wsi_mval.state_dict(),
                        'mlp_cond2_pers_hist_loop_norm_mha_dec':mlp_cond2_pers_hist_loop_norm_mha_dec.state_dict(),
                        'mlp_cond2_pers_hist_loop_norm_mha_dec_cat':mlp_cond2_pers_hist_loop_norm_mha_dec_cat.state_dict(),

                        'mlp_token_cp_pers_cc':mlp_token_cp_pers_cc.state_dict(),
                        'mlp_token_cp_pers_cc_norm':mlp_token_cp_pers_cc_norm.state_dict(),
                        'mlp_token_cp_pers_cc_binary':mlp_token_cp_pers_cc_binary.state_dict(),
                        'mlp_token_cp_pers_loop':mlp_token_cp_pers_loop.state_dict(),
                        'mlp_token_cp_pers_loop_norm':mlp_token_cp_pers_loop_norm.state_dict(),
                        'mlp_token_cp_pers_loop_binary':mlp_token_cp_pers_loop_binary.state_dict(),

                        'optimizer':optimizer.state_dict(),
                        'optimizer_aux':optimizer_aux.state_dict(),
                        }
        saved = False
        loss_new = cond2_pers_hist_loop_norm_cumsum_loss + cond2_pers_hist_cc_norm_cumsum_loss \
            + cond2_pers_hist_cc_norm_reg_loss + cond2_pers_hist_loop_norm_reg_loss \
            + token_cp_pers_cc_norm_loss + token_cp_pers_loop_norm_loss \
            + token_cp_pers_cc_binary_loss  + token_cp_pers_loop_binary_loss

        # if(loss_new <min_val_loss ):
        #     min_val_loss = loss_new
        #     if(not saved):
        #         torch.save(state_dict, os.path.join(checkpoints_save_path, f"toposlide_epoch_{epoch}_min.pt"))
        #         saved = True

        # if ((epoch % model_save_freq == 0  and epoch > start_save_epoch) or epoch == args.num_epochs - 1 or epoch >= args.num_epochs-10 or min_total_loss_val >= epoch_val_loss):
        if ((epoch % model_save_freq == 0  and epoch > start_save_epoch) or epoch == args.num_epochs - 1):

            if(val_e_indx < 0):
                val_loss_smoothed_epoch_arr = np.arange(epoch, args.num_epochs + model_save_freq, model_save_freq).astype(int)
                val_loss_smoothed_arr = np.zeros(len(val_loss_smoothed_epoch_arr))

            val_e_indx += 1

            val_loss_smoothed_epoch_arr[val_e_indx] = int(epoch)
            if(val_e_indx==0):
                val_loss_smoothed_arr[val_e_indx] = loss_new
            else:
                val_loss_smoothed_arr[val_e_indx] = ema_alpha * loss_new + (1-ema_alpha)*val_loss_smoothed_arr[-1]
                x_points =val_loss_smoothed_epoch_arr
                y_points = val_loss_smoothed_arr
                poly5 = np.polyfit(x_points, y_points, 5)
                f = np.poly1d(poly5)
                x_new = np.linspace(x_points[0], x_points[-1], x_points[-1]-x_points[0]+1)
                y_new = f(x_new)
                grad_smoothed = np.gradient(y_new, x_new) 
                val_loss_smoothed_grad_arr[val_e_indx] = grad_smoothed[-1]
                print('val_loss_smoothed_grad_arr[val_e_indx]', val_loss_smoothed_grad_arr[val_e_indx], 'epoch', epoch)

                if(val_loss_smoothed_grad_arr[val_e_indx]<0 and abs(val_loss_smoothed_grad_arr[val_e_indx])<abs(grad_smoothed_min)):
                    grad_smoothed_min = val_loss_smoothed_grad_arr[val_e_indx]
                    epoch_min = val_loss_smoothed_epoch_arr[val_e_indx]
                    print('grad_smoothed min', grad_smoothed_min, 'epoch', epoch_min)
                    best_epoch = epoch
                    if(not saved):
                        torch.save(state_dict, os.path.join(checkpoints_save_path, f"toposlide_epoch_{epoch}.pt"))
                        saved = True
                elif(val_loss_smoothed_grad_arr[val_e_indx] > 0.0003 or (best_epoch>=0 and epoch-best_epoch > patience)):
                    early_stop_flag = True

                # if(loss_new <min_val_loss ):
                #     min_val_loss = loss_new
                #     if(not saved):
                #         torch.save(state_dict, os.path.join(checkpoints_save_path, f"toposlide_epoch_{epoch}.pt"))
                #         saved = True

                torch.save(state_dict, os.path.join(checkpoints_save_path, f"toposlide_last.pt"))

