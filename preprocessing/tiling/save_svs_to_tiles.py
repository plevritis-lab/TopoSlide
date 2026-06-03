'''
    Adapted from https://github.com/ShahiraAbousamra/til_classification
'''

import sys
import os
import argparse
import glob
import traceback

import numpy as np
import openslide
from PIL import Image
import h5py
import pandas as pd
import skimage.io as io

'''
python save_svs_to_tiles.py \
    --svs_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad/svs" \
    --patches_out_root_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad/patches_20x_512" \
    --target_mag 20 \
    --patch_size_at_target_mag 512 \
    --default_mpp 0.254 \
    --wsi_ext ".svs" \
    --store_hdf5 \
'''


lock_filename = "processing.txt"
done_filename = "done.txt"
error_filename = "error.txt"


def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Tile a WSI into patches")    
    parser.add_argument("--name", type=str, default='tiling')
    parser.add_argument("--svs_dir", type=str)
    parser.add_argument("--patches_out_root_dir", type=str)
    # parser.add_argument("--log_out_dir", type=str)
    parser.add_argument("--target_mag", type=int, default=20)
    parser.add_argument("--patch_size_at_target_mag", type=int, default=512)
    parser.add_argument("--default_mpp", type=float, default=0.254)
    parser.add_argument("--wsi_ext", type=str, default=".svs")
    parser.add_argument("--store_hdf5", action='store_true')

    args = parser.parse_args()

    svs_dir = args.svs_dir
    patches_out_root_dir = args.patches_out_root_dir
    # log_out_dir = args.log_out_dir
    target_mag = args.target_mag
    patch_size_at_target_mag = args.patch_size_at_target_mag
    store_hdf5 = args.store_hdf5
    default_mpp = args.default_mpp
    wsi_ext = args.wsi_ext

    if(not os.path.exists(patches_out_root_dir)):
        os.makedirs(patches_out_root_dir, exist_ok=True)

    svs_files = glob.glob(os.path.join(svs_dir, f"*{wsi_ext}"))

    for indx in np.random.permutation(len(svs_files)):
        try:
            svs_filepath = svs_files[indx]
            svs_filename = os.path.basename(svs_filepath)

            slide_name = os.path.basename(svs_filename).split('.')[0]
            if(not store_hdf5):
                output_folder = patches_out_root_dir + '/' + svs_filename;
            else:
                output_folder = patches_out_root_dir

            lock_filepath = os.path.join(patches_out_root_dir, f"{slide_name}_{lock_filename}")
            done_filepath = os.path.join(patches_out_root_dir, f"{slide_name}_{done_filename}")
            error_filepath = os.path.join(patches_out_root_dir, f"{slide_name}_{error_filename}")

            if os.path.isfile(done_filepath):
                print('fdone {} exist, skipping'.format(done_filepath));
                continue

            if(os.path.exists(lock_filepath) or os.path.exists(done_filepath)):
                continue
            with open(lock_filepath, 'w') as lock_file:
                lock_file.write('processing')

            print('extracting {}'.format(svs_filename));

            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True);

            #create HDF5 file
            if(store_hdf5):
                hdf5_filename = f"{slide_name}_{int(target_mag)}x_{int(patch_size_at_target_mag)}.hdf5"
                # hdf5_filename = f"{slide_name}_{target_mag}x_{patch_size_at_target_mag}_gzip.hdf5"
                # hdf5_filename = f"{slide_name}_{target_mag}x_{patch_size_at_target_mag}_gzip9.hdf5"
                # hdf5_filename = f"{slide_name}_{target_mag}x_{patch_size_at_target_mag}_gzip_chunked.hdf5"
                # hdf5_filename = f"{slide_name}_{target_mag}x_{patch_size_at_target_mag}_gzip_index.hdf5"    
                hf = h5py.File(os.path.join(output_folder,hdf5_filename), 'w')
                meta_filename = f"{slide_name}_{int(target_mag)}x_{int(patch_size_at_target_mag)}_meta.csv"
                meta_df = pd.DataFrame()

            try:
                oslide = openslide.OpenSlide(svs_filepath);
                # print('oslide.properties', oslide.properties)
                # for k,v in oslide.properties.items():
                #     print(k,v)
            #    mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
                if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
                   mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
                   # print("oslide.properties[openslide.PROPERTY_NAME_MPP_X]", oslide.properties[openslide.PROPERTY_NAME_MPP_X])
                # elif "XResolution" in oslide.properties:
                #    mag = 10.0 / float(oslide.properties["XResolution"]);
                #    print("float(oslide.properties[XResolution]", float(oslide.properties["XResolution"]))
                # elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
                #    mag = 10.0 / float(oslide.properties["tiff.XResolution"]);
                #    print("float(oslide.properties[XResolution]", float(oslide.properties["tiff.XResolution"]))
                else:
                   mag = 10.0 / float(default_mpp);
                   # print("default", float(default_mpp))
                pw = int(patch_size_at_target_mag * mag / target_mag);
                width = oslide.dimensions[0];
                height = oslide.dimensions[1];
            except:
                print('{}: exception caught'.format(slide_name));
                try:
                    oslide = None
                    slide_im = Image.open(svs_filepath)
                    mag = 10.0 / float(default_mpp);
                    pw = int(patch_size_at_target_mag * mag / target_mag);
                    width = slide_im.width 
                    height = slide_im.height 
                except:
                    print('{}: exception caught'.format(slide_name));
                    exit(1);


            print(slide_name, width, height);

            index = 0
            patch_filenames = []
            x_coord = []
            y_coord = []
            whiteness_list = []
            print("pw", pw)
            for x in range(1, width, pw):
                for y in range(1, height, pw):
                    if x + pw > width:
                        pw_x = width - x;
                    else:
                        pw_x = pw;
                    if y + pw > height:
                        pw_y = height - y;
                    else:
                        pw_y = pw;

                    if (int(patch_size_at_target_mag * pw_x / pw) <= 0) or \
                       (int(patch_size_at_target_mag * pw_y / pw) <= 0) or \
                       (pw_x <= 0) or (pw_y <= 0):
                        continue;

                    if(oslide):
                        patch = oslide.read_region((x, y), 0, (pw_x, pw_y));
                        #shahira: skip where the alpha channel is zero
                        patch_arr = np.array(patch);
                        if(patch_arr[:,:,3].max() == 0):
                            continue;
                    else:
                        patch = slide_im.crop((x, y, x+pw_x, y+pw_y))
            
                    # Resize into 20X.
                    patch = patch.resize((int(patch_size_at_target_mag * pw_x / pw), int(patch_size_at_target_mag * pw_y / pw)), Image.LANCZOS );
                    if(not store_hdf5):
                        fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_at_target_mag);
                        patch.save(fname);
                    else:
                        patch_name = '{}_{}_{}_{}.png'.format(x, y, pw, patch_size_at_target_mag)
                        patch_filenames.append(patch_name)
                        x_coord.append(x)
                        y_coord.append(y)
                        whiteness_list.append(whiteness(np.array(patch)))
                        dataset_name = str(index)
                        # hf.create_dataset(dataset_name, data = np.asarray(patch, dtype=np.uint8), dtype=np.uint8)
                        hf.create_dataset(dataset_name, data = np.asarray(patch, dtype=np.uint8), dtype=np.uint8, compression="gzip")
                        # hf.create_dataset(dataset_name, data = np.asarray(patch, dtype=np.uint8), dtype=np.uint8, compression="gzip", compression_opts=9) # no improvement with compression level 9 but time is multiplied
                        # hf.create_dataset(dataset_name, data = np.asarray(patch, dtype=np.uint8), dtype=np.uint8, compression="gzip", chunks=True) # no effect because each dataset is a single image
                    index += 1

            if(store_hdf5):
                hf.close()
                meta_df['indexing'] = np.arange(index)
                meta_df['patch_name'] = np.array(patch_filenames)
                meta_df['x_coord'] = np.array(x_coord)
                meta_df['y_coord'] = np.array(y_coord)
                meta_df['whiteness'] = np.array(whiteness_list)
                meta_df.to_csv(os.path.join(output_folder, meta_filename),index=False)

            with open(done_filepath, 'w') as done_file:
                done_file.write('done')

        except Exception as e:
            print(e)
            traceback.print_exc()
            with open(error_filepath, 'w') as error_file:
                error_file.write('error\n')
                error_file.write(str(e))
                error_file.write("\n")
                error_file.write(traceback.format_exc())
                error_file.write("\n")
                error_file.flush()
            continue
