import sys
import os
import glob
import traceback
import argparse

import numpy as np
import h5py
import pandas as pd
import skimage.io as io
import cv2


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Save to HDF5")    

    parser.add_argument("--root_input_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp")
    parser.add_argument("--out_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo")

    args = parser.parse_args()

    root_dir = args.root_input_dir
    out_dir = args.out_dir

    error_filename = "error.txt"

    if(not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    slides = os.listdir(root_dir)

    # for slide_name in slides:
    for wi in np.random.permutation(len(slides)):
        slide_name = slides[wi]
        try:
            print(wi, slide_name, end='\r')
            slide_folder = os.path.join(root_dir, slide_name)
            if(not os.path.isdir(slide_folder)):
                continue
            error_filepath = os.path.join(out_dir, f"{slide_name}_{error_filename}")
            hdf5_filename = f"{slide_name}_topo.hdf5"
            hdf5_filepath = os.path.join(out_dir, hdf5_filename)
            meta_filename = f"{slide_name}_topo_meta.csv"
            meta_filepath = os.path.join(out_dir, meta_filename)
            if(os.path.exists(hdf5_filepath) or os.path.exists(meta_filepath)):
                continue

            hf = h5py.File(hdf5_filepath, 'w')
            meta_df = pd.DataFrame()
    
            files = glob.glob(os.path.join(slide_folder, "*.*"))
            index = 0
            topo_filenames = []
            for filepath in files:
                if(not os.path.isfile(filepath)):
                    continue
                ext = os.path.splitext(filepath)[-1]
                dataset_name = os.path.basename(filepath)
                if(ext == '.png'):
                    try:
                        im = io.imread(filepath)
                    except Exception as e1:
                        print('Exception', e1)
                        print(traceback.format_exc())
                        try:
                            im = cv2.imread(filepath)
                            print(im.shape)
                        except Exception as e2:
                            print('Exception', e2)
                            print(traceback.format_exc())
                            raise e2
                    hf.create_dataset(dataset_name, data = np.asarray(im, dtype=np.uint8), dtype=np.uint8, compression="gzip")
                elif(ext == '.npy'):
                    arr = np.load(filepath, allow_pickle=True)
                    hf.create_dataset(dataset_name, data = np.asarray(arr, dtype=arr.dtype), dtype=arr.dtype, compression="gzip")
                elif(ext == '.csv'):
                    df = pd.read_csv(filepath)
                    group = hf.create_group(dataset_name)
                    # Create datasets for the DataFrame's data and index
                    group.create_dataset('data', data=df.values)
                    group.create_dataset('index', data=df.index.values)
                    group.create_dataset('columns', data=df.columns.values.astype('S')) 
                    # reading
                    # with h5py.File('data.h5', 'r') as f:
                    #    group = f['my_dataframe']
                    #    df_read = pd.DataFrame(group['data'][:], index=group['index'][:], columns=group['columns'][:].astype(str))
                topo_filenames.append(dataset_name)
                index += 1
            hf.close()
            meta_df['indexing'] = np.arange(index)
            meta_df['key_name'] = np.array(topo_filenames)
            meta_df.to_csv(meta_filepath,index=False)
            # break
        except Exception as e:
            print('Exception', e)
            print(traceback.format_exc())
            with open(error_filepath, 'w') as error_file:
                error_file.write('error\n')
                error_file.write(str(e))
                error_file.write("\n")
                error_file.write(traceback.format_exc())
                error_file.write("\n")
                error_file.flush()
            continue
        sys.stdout.flush()
