import os
import sys
import glob

import numpy as np
import pandas as pd

# out_dir = "/oak/stanford/groups/plevriti/shahira/datasets/tcga_lung_lusc_data_splits/train_test_5fold_site_preserving2"
out_dir = "/oak/stanford/groups/plevriti/shahira/datasets/tcga_brca/data_splits/train_test_5fold_site_preserving2"

# svs_root_dir = "/oak/stanford/groups/plevriti/shahira/tcga_luad_svs/all"
svs_root_dir_list = ["/oak/stanford/groups/plevriti/shahira/datasets/tcga_brca/svs",
                     ]

n_splits = 5

if __name__ == "__main__":

    if(not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    svs_filepaths_list = []
    for svs_root_dir in svs_root_dir_list:
        svs_filepaths = glob.glob(os.path.join(svs_root_dir, "*.svs"))
        svs_filepaths_list.append(svs_filepaths)
    svs_filepaths = np.concatenate(svs_filepaths_list)
    print("svs_filepaths", len(svs_filepaths))
    # print(svs_filepaths)


    # Extract site from tcga filepath
    site_name_list = []
    slide_name_list = []
    
    for svs_file in svs_filepaths:
        slide_name = os.path.splitext(os.path.basename(svs_file))[0].split('.')[0]
        site_name = slide_name.split('-')[1]
        slide_name_list.append(slide_name)
        site_name_list.append(site_name)
        # print(slide_name)
        # print(site_name)
    slide_name_arr = np.array(slide_name_list)
    site_name_arr = np.array(site_name_list)
    unique_site_names = np.unique(site_name_arr)    
    print('unique_site_names')
    print(unique_site_names)
    permuted_sites = np.random.permutation(unique_site_names)
    site_counts = np.zeros(len(permuted_sites))
    for i, site in enumerate(permuted_sites):
        site_counts[i] = (site_name_arr==site).sum()
    print('permuted_sites')
    print(permuted_sites )
    fold_size = len(slide_name_list)//n_splits
    print('len(slide_name_list)', len(slide_name_list))
    print('fold_size', fold_size)

    train_lists = [[] for i in range(n_splits)]
    test_lists = [[] for i in range(n_splits)]
    train_counts = np.zeros((n_splits), dtype=int)
    test_counts = np.zeros((n_splits), dtype=int)

    for si in range(len(permuted_sites)):
        ci = site_counts[si]
        print(si,permuted_sites[si], ci)
        ki = test_counts.argmin()
        # print('test_counts.argmin()', ki)
        # print('slide_name_arr[site_name_arr == permuted_sites[si]]\n', slide_name_arr[site_name_arr == permuted_sites[si]])
        test_lists[ki].append(slide_name_arr[site_name_arr == permuted_sites[si]])
        test_counts[ki] += ci
        # print('test_lists[ki]', test_lists[ki])
        # print('test_counts[ki]', test_counts[ki])
        # break
    for ki in range(n_splits):
        test_lists[ki] = np.concatenate(test_lists[ki], axis=0)
        # print(f'test_lists[{ki}]')
        # print(test_lists[ki])
        # print(len(test_lists[ki]))

    
    for ti in range(n_splits):
        for ki in range(n_splits):
            if(ti == ki):
                continue
            train_lists[ti].append(test_lists[ki])
        train_lists[ti] = np.concatenate(train_lists[ti], axis=0)
        train_counts[ti] = len(train_lists[ti])
        df_train = pd.DataFrame({'slide_ids':train_lists[ti]})
        df_test = pd.DataFrame({'slide_ids':test_lists[ti]})
        df_train.to_csv(os.path.join(out_dir, f'train_{ti+1}.csv'),index=False)
        df_test.to_csv(os.path.join(out_dir, f'test_{ti+1}.csv'),index=False)
        print('split', ti, 'train', len(train_lists[ti]), 'test', len(test_lists[ti]), 'sum', len(train_lists[ti])+len(test_lists[ti]))

    splits_stats_df = pd.DataFrame({'split_id':np.arange(1,n_splits+1),'train_count':train_counts, 'test_count':test_counts})
    splits_stats_df.to_csv(os.path.join(out_dir, f'splits_stats.csv'),index=False)





