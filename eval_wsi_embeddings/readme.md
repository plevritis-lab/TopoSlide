
### Generate WSI Embeddings


1. Generate WSI embeddings using all the foreground patches as input to TopoSlide: call ```run_toposlide_eval.py```. \

- Check **script** used for generating WSI embeddings for the TCGA_LUAD dataset: ```scripts\run_eval_wsi_embeddings_tcga_luad.sh```  
	
-	Input parameters are:  \
	**patch_embedding_dir:** Input directory containing patch embeddings. For each slide, a file named ```<slide_name><*>.h5``` contains ```features``` and ```coords```.  \
	**slide_root_output_dir:** Output directory.  \
	**checkpoint_path:** Model filepath.  \
	**patch_size_lv0:** The patch size. Default is 512.  \

- TopoSlide WSI embeddings for TCGA_LUAD are available at: ```datasets/tcga_luad/patches_20x_512_toposlide_wsi_embedding```  

- TCGA_LUAD meta data is available at: ```datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv```  

- TCGA_LUAD clustering centroids are available at: ```datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/kmeans_clustering_conch512_n16_centers.npy```  

- Other samples for TCGA_LUAD tiling, patch embeddings, and clustering are available under: ```datasets/tcga_luad```  
   <br>
  
  
  
2. Generate WSI embeddings using all the foreground patches, while excluding patches belonging to some clusters (ex. representing artifacts and other irrelevant or background regions), as input to TopoSlide, call ```run_toposlide_eval_w_exclude.py```. \

-	Check **script** used for generating WSI embeddings for the DHMC_LUAD dataset: ```scripts\run_eval_wsi_embeddings_dhmc_luad.sh```  

-	Input parameters are:  \
	**patch_embedding_dir:** Input directory containing patch embeddings. For each slide, a file named ```<slide_name><*>.h5``` contains ```features``` and ```coords```.  \
	**slide_root_output_dir:** Output directory.  \
	**checkpoint_path:** Model filepath.  \
	**patch_size_lv0:** The patch size. Default is 512.  \
	**wsi_dim_filepath:** Input csv file contains meta data for the slide. It has the columns: ```slide_name```, ```slide_width```,	```slide_height```,	```slide_mag```, ```slide_pw```, where ```slide_pw``` is the tile size at the highest magnification. \
	**clustering_dir:** Input directory containing the clustering of patches in each WSI. For each slide, a file named ```<slide_name><*>.csv``` contains the following columns: ```coord_x```, ```coord_y```, ```cluster_id``` \
	**ignore_cluster_ids:** A list of cluster IDs to exclude their patches when computing the WSI embedding. This is useful for excluding artifacts and other irrelevant or background regions in the slide.\
	**n_clusters:** The number of clusters (i.e. max cluster ID + 1). Default is 16.  \

- TopoSlide WSI embeddings for DHMC_LUAD are available at: ```datasets/dhmc_luad/patches_20x_512_toposlide_wsi_embedding```  

- DHMC_LUAD meta data is available at: ```datasets/dhmc_luad/dhmc_luad_wsi_dim_p512_m20.csv```  



