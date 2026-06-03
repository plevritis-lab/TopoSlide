
### WSI Tiling

Convert WSIs into patches.  


#### Run the tiling:
```
cd TopoSlide/preprocessing/tiling 
nohup python python save_svs_to_tiles.py \
    --svs_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad/svs" \
    --patches_out_root_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad/patches_20x_512" \
    --log_out_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad/patches_20x_512_log" \
    --target_mag 20 \
    --patch_size_at_target_mag 512 \
    --default_mpp 0.254 \
    --wsi_ext ".svs" \
    --store_hdf5 &
```
The command can be run multiple times for parallel processing.

