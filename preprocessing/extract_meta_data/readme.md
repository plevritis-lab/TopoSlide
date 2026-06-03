
### Extract WSI Meta Data

Extract meta data including: slide width, height, original magnification, and original patch size given a target size at a target magnification.  


1. ```extract_wsi_dim.py```  
- Creates a csv file with the following filename format:  
 ```<dataset_name>_wsi_dim_p<patch_size_at_target_mag>_m<target_mag>.csv```  
ex. ```tcga_luad_wsi_dim_p512_m20.csv```     
with the following columns: ```slide_name, slide_width, slide_height, slide_mag, slide_pw```.  

- Check **script** used for generating WSI embeddings for the TCGA_LUAD dataset: ```scripts/run_extract_wsi_meta_tcga_luad.sh```  
	
-	Input parameters are:  \
	**svs_dir:** Input directory containing WSI files.  \
	**out_dir:** Output directory.  \
	**out_filename:** Output filename.  \
	**slide_ext:** The WSI files extenstion, ex. svs or tif. Default is svs.  \
	**default_original_mpp:** If microns per pixel not in openslide property ```MPP_X```, use this value. Default is 0.254 which corresponds to 40x magnification.  \
	**target_mag:** The target magnification for patch extraction. Default is 20.  \
	**patch_size_at_target_mag:** The desired patch size at that magnification. Default is 512.  

- Sample meta data file is available at: ```datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv```  


2. ```extract_wsi_dim_dhmc.py```  
- Same as ```extract_wsi_dim.py``` but customized to handle the DHMC_LUAD dataset, where the default MPP is read from an external file.  

- Check **script** used for generating WSI embeddings for the DHMC_LUAD dataset: ```scripts/run_extract_wsi_meta_dhmc_luad.sh```  
	
-	Additional parameters are:  \
	**svs_meta_filepath:** The filepath of the meta data file provided with the dataset. It is a csv file includes the following columns: ```File Name``` and ```Microns Per Pixel```.  \

- Sample meta data file is available at: ```datasets/dhmc_luad/dhmc_luad_wsi_dim_p512_m20.csv```  
  
  
