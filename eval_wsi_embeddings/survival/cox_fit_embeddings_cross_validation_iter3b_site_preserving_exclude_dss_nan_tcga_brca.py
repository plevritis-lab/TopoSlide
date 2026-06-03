import os
import sys
import glob
import traceback
import argparse
import pickle

import numpy as np
import pandas as pd
import io
import torch
import h5py
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import LabelEncoder
from Levenshtein import distance as lev_dist
from Levenshtein import hamming
from sksurv.linear_model import CoxPHSurvivalAnalysis
# from sklearn.model_selection import KFold

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# embedding_dir = "/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/patches_20x_512_titan_slide_embedding"

# out_root_dir = "/oak/stanford/groups/plevriti/shahira/wsi_rep_eval_results/survival_reg/luad"

# method_name = "titan_original"

# # embedding_key = "pretrained_original"

# survival_filepath = f"/oak/stanford/groups/plevriti/shahira/TCGA-CDR-SupplementalTableS1_luad_cdr.csv"

# wsi_dim_filepath = "/oak/stanford/groups/plevriti/shahira/topo_analysis_results_luad_all/tcga_luad_wsi_slide_sizes5_tmp/tcga_luad_wsi_comp_largest_tumor.csv"

splits_folder = f"/oak/stanford/groups/plevriti/shahira/datasets/tcga_brca/data_splits/train_test_5fold_site_preserving2"
train_split_filename_pattern = "train_<ki>.csv"
test_split_filename_pattern = "test_<ki>.csv"

# split_ratio = np.array([0.6,0.2,0.2]) # train, val, test
n_splits = 5



def load_csv_as_df(filepath, columns=None):
    if(columns is None):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, header=0, usecols=columns)
    return df

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Survival Cox Reg")    
    parser.add_argument("--embedding_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/datasets/tcga_brca/patches_20x_512_titan_slide_embedding")
    parser.add_argument("--out_root_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/wsi_rep_eval_results/survival_reg/tcga_brca")
    parser.add_argument("--method_name", type=str, default="titan_original")
    parser.add_argument("--survival_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/TCGA-CDR-SupplementalTableS1_brca_cdr.csv")
    parser.add_argument("--splits_folder", type=str, default="/oak/stanford/groups/plevriti/shahira/datasets/tcga_brca/data_splits/train_test_5fold_site_preserving2")
    parser.add_argument("--wsi_dim_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/tcga_brca_wsi_dim.csv")
    parser.add_argument("--task", type=str, default="cox_5fold_iter3b")

    args = parser.parse_args()

    embedding_dir = args.embedding_dir
    out_root_dir = args.out_root_dir
    method_name = args.method_name
    survival_filepath = args.survival_filepath
    splits_folder = args.splits_folder
    wsi_dim_filepath = args.wsi_dim_filepath
    task = args.task
    
    out_dir = os.path.join(out_root_dir, f"{task}")
    if(not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    
    wsi_dim_df = pd.read_csv(wsi_dim_filepath)
    wsi_dim_arr = wsi_dim_df.to_numpy()

    # Get the slide embeddings
    # slide_embedding_files = glob.glob(os.path.join(embedding_dir, "*.h5"))
    slide_embedding_files = glob.glob(os.path.join(embedding_dir, "*.pkl"))




    print('len(slide_filepath)', len(slide_embedding_files))


    slide_embeddings_arr = None

    slide_names_list = []
    slide_ids_main = []
    wi = -1
    for slide_filepath in slide_embedding_files:
        slide_name = os.path.splitext(os.path.basename(slide_filepath))[0].split('_')[0]
        try:
            wi_stats_df = wsi_dim_df[wsi_dim_df['slide_name']==slide_name]
            if(len(wi_stats_df)==0):
                continue
            # slide_hf = h5py.File(slide_filepath, 'r')
            try:
                with open(slide_filepath, 'rb') as file:
                    embedding = CPU_Unpickler(file).load()
            except:
                print("error loading slide embedding from", slide_filepath)
                continue            
            wi += 1
        except:
            print("error loading slide embedding from", slide_filepath)
            continue
        # print('embedding', embedding.shape)
        # print('slide_name', slide_name)
        if(slide_embeddings_arr is None):
            slide_embeddings_arr = np.zeros((len(slide_embedding_files), embedding.shape[-1]))
        slide_embeddings_arr[wi] = embedding
        slide_names_list.append(slide_name)
        slide_ids_main.append(slide_name[:len("TCGA-49-4488")])
        # print(slide_name)
        print(wi, slide_name[:len("TCGA-49-4488")], end='\r')
        # print(wi, slide_name[:len("TCGA-49-4488")])
        # break
    slide_embeddings_arr = slide_embeddings_arr[:len(slide_names_list)]
    # slide_names_arr = np.array(slide_names_list)
    # print('slide_names_arr\n', slide_names_arr)

    print('slide_embeddings_arr', slide_embeddings_arr.shape)
    print('slide_names_list', len(slide_names_list))


    survival_df = load_csv_as_df(survival_filepath, columns=['bcr_patient_barcode', 'OS.time', 'DSS.time', 'PFI', 'PFI.time', 'DSS'])
    survival_df['OS.time_months'] = survival_df['OS.time']/30
    survival_df['DSS.time_months'] = survival_df['DSS.time']/30
    survival_df['PFI.time_months'] = survival_df['PFI.time']/30
    # survival_df = survival_df[pd.notnull(survival_df["PFI.time"])] # exclude nan
    survival_df = survival_df[pd.notnull(survival_df["DSS.time"])] # exclude nan
    survival_df = survival_df[pd.notnull(survival_df["DSS"])] # exclude nan
    survival_slide_ids = survival_df['bcr_patient_barcode'].to_numpy()

    slide_ids = np.array(slide_names_list)
    # slide_ids_main = np.array([s[:s.find('-01Z-')] for s in slide_ids]).astype(str)   
    slide_ids_main = np.array(slide_ids_main)


    topo_data_rows_indx = []
    survival_slide_ids_new = []
    surv_censored = []
    surv_dss = []
    surv_os = []
    surv_pfi = []
    slides_missing_he = []
    slides_missing_surv = []
    for slide_id in survival_slide_ids:
        if((slide_ids_main==slide_id).sum() > 0):
            for wi in range((slide_ids_main==slide_id).sum()):
                topo_data_rows_indx.append(np.where(slide_ids_main==slide_id)[0][wi])
                survival_slide_ids_new.append(slide_id)
                slide_surv_df = survival_df[survival_df['bcr_patient_barcode']==slide_id]
                surv_censored.append(slide_surv_df['DSS'].values[0])
                surv_dss.append(slide_surv_df['DSS.time_months'].values[0])
                surv_os.append(slide_surv_df['OS.time_months'].values[0])
                surv_pfi.append(slide_surv_df['PFI.time_months'].values[0])
        else:
            slides_missing_he.append(slide_id)

    for slide_id in slide_ids_main:
        if((survival_slide_ids==slide_id).sum() == 0):
            slides_missing_surv.append(slide_id)

    print('slides_missing_he', len(slides_missing_he))
    # for slide_id in slides_missing_he:
    #     print(slide_id)
    # print("\n\n")
    print('slides_missing_surv', len(slides_missing_surv))
    for slide_id in slides_missing_surv:
        print(slide_id)

    survival_slide_ids_new = np.array(survival_slide_ids_new)
    topo_data_rows_indx = np.array(topo_data_rows_indx)
    print('topo_data_rows_indx', topo_data_rows_indx.shape)
    surv_censored = np.array(surv_censored)
    surv_dss = np.array(surv_dss)
    surv_os = np.array(surv_os)
    surv_pfi = np.array(surv_pfi)

    # surv_dss_y = list(zip(surv_censored, surv_dss))
    surv_dss_y = np.array(list(zip(surv_censored.astype(bool), surv_dss)), dtype=[('Event_indicator', bool), ('Time', float)])
    print('surv_dss_y', surv_dss_y.shape)

    # alphas = 10.0 ** np.linspace(-4, 4, 50)
    alphas = 10.0 ** np.linspace(1, 5, 25)
    alphas = np.concatenate(([0],alphas))
    # vis_input_dir = os.path.join(vis_input_root_dir, topo_map_name)
    
    # perm = np.random.permutation(len(topo_data_rows_indx))
    # split_size = split_ratio * len(topo_data_rows_indx)
    X_dummy = np.zeros((len(topo_data_rows_indx), 1))

    # StratifiedKFold  # when there are multiple classes like luad and lusc
    # from sklearn.model_selection import StratifiedKFold
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    
    # kf = KFold(n_splits=n_splits)
    # kf_out = kf.get_n_splits(X_dummy)
    
    # fold_size = len(topo_data_rows_indx)//n_splits
    # val_start_indx = 0




    with open(os.path.join(out_dir, f"{method_name}_cox_fit_b.csv"), "w") as out_file:
        out_file.write("feature_name,alpha")
        for ki in range(n_splits):
            out_file.write(f",split{ki},train_cindex_dss{ki},test_cindex_dss{ki}")
        out_file.write(f",mean,std")
        out_file.write("\n")
        out_file.flush()

        surv_data_x = slide_embeddings_arr[topo_data_rows_indx]
        print("np.isnan(surv_data_x).sum()", np.isnan(surv_data_x).sum())
        print("np.isnan(surv_dss).sum()", np.isnan(surv_dss).sum())
        print("np.isnan(surv_censored).sum()", np.isnan(surv_censored).sum())

        # clustering_name = "gigapath_pretrained_original"
        row_test_cindex = np.zeros(n_splits)
        for alpha in alphas:
            print('alpha', alpha)
            out_file.write(f"{method_name},{alpha}")
            out_file.flush()
            # ki = 0
            val_start_indx = 0
            # kf_out = kf.get_n_splits(X_dummy)
            # print('kf_out', kf_out)

            # # for train_indices0, test_indices0 in kf_out:
            # for ki, (train_indices0, test_indices0) in enumerate(kf.split(X_dummy)):

            for ki in range(n_splits):
                train_slide_names = pd.read_csv(os.path.join(splits_folder, train_split_filename_pattern.replace('<ki>', str(ki+1))))['slide_ids'].values
                test_slide_names = pd.read_csv(os.path.join(splits_folder, test_split_filename_pattern.replace('<ki>', str(ki+1))))['slide_ids'].values
                # train_slide_ids_main = np.array([s[:s.find('-01Z-')] for s in train_slide_names]).astype(str)   
                # test_slide_ids_main = np.array([s[:s.find('-01Z-')] for s in test_slide_names]).astype(str)   
                train_slide_ids_main = np.array([s[:len("TCGA-49-4488")] for s in train_slide_names]).astype(str)   
                test_slide_ids_main = np.array([s[:len("TCGA-49-4488")] for s in test_slide_names]).astype(str)   

                try:
                    print('ki', ki)
                    # val_indices = perm[train_indices0[val_start_indx:val_start_indx+fold_size]]
                    # train_indices = perm[np.concatenate((train_indices0[0:val_start_indx], train_indices0[val_start_indx+fold_size:]))]
                    # val_start_indx = (val_start_indx + fold_size)%len(train_indices0)
                    # test_indices = perm[test_indices0]
                    # # print('train0', train_indices0)
                    # # print('train', train_indices)
                    # # print('val_indices', val_indices)
                    # # print('test_indices', test_indices)

                    # print('survival_slide_ids_new', survival_slide_ids_new)
                    # print('test_slide_names', test_slide_names)

                    train_slides, train_indices0, train_indices1= np.intersect1d(survival_slide_ids_new, train_slide_ids_main, return_indices=True)
                    test_slides, test_indices0, test_indices1= np.intersect1d(survival_slide_ids_new, test_slide_ids_main, return_indices=True)
                    print('train_indices0', len(train_indices0))
                    print('test_indices0', len(test_indices0))

                    surv_data_x_train = surv_data_x[train_indices0]
                    # surv_data_x_val = surv_data_x[val_start_indx]
                    surv_data_x_test = surv_data_x[test_indices0]

                    surv_dss_y_train = surv_dss_y[train_indices0]
                    # surv_dss_y_val = surv_dss_y[val_start_indx]
                    surv_dss_y_test = surv_dss_y[test_indices0]

                    estimator_dss = CoxPHSurvivalAnalysis(alpha=alpha,n_iter=3).fit(surv_data_x_train, surv_dss_y_train)
                    cindex_dss_train = estimator_dss.score(surv_data_x_train, surv_dss_y_train)
                    # cindex_dss_val = estimator_dss.score(surv_data_x_val, surv_dss_y_val)
                    cindex_dss_test = estimator_dss.score(surv_data_x_test, surv_dss_y_test)
                    row_test_cindex[ki] = cindex_dss_test

                    out_file.write(f",{ki},{cindex_dss_train},{cindex_dss_test}")
                    out_file.flush()
                    # ki += 1

                except Exception as e:
                    # print(e)
                    row_test_cindex[ki] = 0
                    out_file.write(f",{ki},0,0")
                    out_file.flush()
                    print('Exception', e)
                    print(traceback.format_exc())
            out_file.write(f",{row_test_cindex.mean()},{row_test_cindex.std()}")
            out_file.write("\n")
            out_file.flush()

