import sys
import os
import glob
import numpy as np
import skimage.io as io
import openslide
import pandas as pd
import argparse

'''
cd TopoSlide/preprocessing/thumbnails/  
nohup python python wsi_2_thumbnails.py \
    --svs_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad_svs/all" \
    --out_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad/thumbnails" \
    --default_mpp 0.254 \
    --target_mag 2 \
    --wsi_ext ".svs" & \
'''

mag_40x_magic_number = 10; # for til magic number is 10 (0.254 mpp) # mag = 10.0 / float(0.254) (gives mag= 40 )


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create thumbnails")    
    parser.add_argument("--name", type=str, default='Create thumbnails')
    parser.add_argument("--svs_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--default_mpp", type=float, default=0.254)
    parser.add_argument("--target_mag", type=float, default=2)
    parser.add_argument("--wsi_ext", type=str, default=".svs")

    args = parser.parse_args()
    svs_dir = args.svs_dir
    out_dir = args.out_dir
    default_mpp = args.default_mpp
    target_mag = args.target_mag
    wsi_ext = args.wsi_ext


    if(not os.path.exists(out_dir )):
        os.makedirs(out_dir, exist_ok=True)

    svs_folder_list = glob.glob(os.path.join(svs_dir, f"*{wsi_ext}"))
    print(len(svs_folder_list))

    for i, filepath in enumerate(svs_folder_list):
        slide_name = os.path.basename(filepath).split('.')[0]
        matching_files = glob.glob(os.path.join(out_dir, f"{slide_name}*.png"))
        if(matching_files is not None and len(matching_files)>0):
            continue
        print(i, 'svs_filename', slide_name)

        oslide = openslide.OpenSlide(filepath)
        width = oslide.dimensions[0];
        height = oslide.dimensions[1];
        # print("opened image")

        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
            mag = mag_40x_magic_number / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
            print("oslide.properties[openslide.PROPERTY_NAME_MPP_X]", oslide.properties[openslide.PROPERTY_NAME_MPP_X])
        # elif "XResolution" in oslide.properties:
        #     mag = mag_40x_magic_number / float(oslide.properties["XResolution"]);
        #     print("oslide.properties[XResolution]", oslide.properties["XResolution"])
        # elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
        #     mag = mag_40x_magic_number / float(oslide.properties["tiff.XResolution"]);
        #     print("oslide.properties[tiff.XResolution]", oslide.properties["tiff.XResolution"])
        else:
            #mag = 10.0 / float(0.254);
            #mag = mag_40x_magic_number / float(0.175) # 0.175 corresponds to 40x in Dan's data
            mag = mag_40x_magic_number / float(default_mpp) 
            print(f"defaulting to mpp {default_mpp}")

        scaling_factor =  target_mag / mag

        # print("scaling_factor", scaling_factor)
        # print("input dim", width, height)
        # print("output dim", width*scaling_factor, height*scaling_factor)
        im = np.array(oslide.get_thumbnail((width*scaling_factor, height*scaling_factor)))
        # print("got thumbnail")
        io.imsave(os.path.join(out_dir, f"{slide_name}_{scaling_factor:0.3f}.png"), im.astype(np.uint8))
        # print("saved")

        # break 



