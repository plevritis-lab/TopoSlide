import sys
import os
import glob
import numpy as np
import openslide
import pandas as pd
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Extract meta data")    

    parser.add_argument("--svs_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/svs")
    parser.add_argument("--out_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad")
    parser.add_argument("--dataset_name", type=str, default="dhmc_luad")
    parser.add_argument("--patch_size_at_target_mag", type=int, default=512)
    parser.add_argument("--target_mag", type=int, default=20)
    parser.add_argument("--slide_ext", type=str, default="tif")
    parser.add_argument("--default_original_mpp", type=float, default=0.5038)
    parser.add_argument("--svs_meta_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/MetaData_Release_1.0.csv")

    svs_meta_filepath="/oak/stanford/groups/plevriti/shahira/datasets/dhmc-luad/MetaData_Release_1.0.csv"

    args = parser.parse_args()

    svs_dir = args.svs_dir
    out_dir = args.out_dir
    patch_size_at_target_mag = args.patch_size_at_target_mag
    target_mag = args.target_mag
    slide_ext = args.slide_ext
    default_original_mpp = args.default_original_mpp
    svs_meta_filepath = args.svs_meta_filepath
    out_filename = f"{args.dataset_name}_wsi_dim_p{patch_size_at_target_mag}_m{target_mag}.csv"

    svs_folder_list = glob.glob(os.path.join(svs_dir,f"*.{slide_ext}"))
    svs_meta_df = pd.read_csv(svs_meta_filepath)

    slide_name_list = []
    slide_width_list = []
    slide_height_list = []
    slide_mag_list = []
    slide_pw_list = []

    for i, filepath in enumerate(svs_folder_list):
        slide_name = os.path.basename(filepath).split('.')[0]
        slide_filename = os.path.basename(filepath)
        print(i, '   svs_filename', slide_name, end='\r')

        oslide = openslide.OpenSlide(filepath)
        width = oslide.dimensions[0];
        height = oslide.dimensions[1];

        slide_meta_record = svs_meta_df[svs_meta_df["File Name"]==slide_filename]    
        mpp = slide_meta_record["Microns Per Pixel"]
        mag = 10.0 / float(mpp);
        # if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        #    mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        # else:
        #    mag = 10.0 / float(default_original_mpp);
        pw = int(patch_size_at_target_mag * mag / target_mag);

        slide_name_list.append(slide_name)
        slide_width_list.append(width)
        slide_height_list.append(height)
        slide_mag_list.append(mag)
        slide_pw_list.append(pw)

        sys.stdout.flush()

    df = pd.DataFrame()
    df['slide_name'] = slide_name_list
    df['slide_width'] = slide_width_list
    df['slide_height'] = slide_height_list
    df['slide_mag'] = slide_mag_list
    df['slide_pw'] = slide_pw_list
    df.to_csv(os.path.join(out_dir, out_filename),index=False)


