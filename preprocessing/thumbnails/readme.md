
### WSI Thumbnails

Convert WSIs into thumbnails.  


#### Run the conversion:
```
cd TopoSlide/preprocessing/thumbnails/  
nohup python python wsi_2_thumbnails.py \
    --svs_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad_svs/all" \
    --out_dir "/oak/stanford/groups/plevriti/shahira/tcga_luad/thumbnails" \
    --default_mpp 0.254 \
    --target_mag 2 \
    --wsi_ext ".svs" \
```
 

