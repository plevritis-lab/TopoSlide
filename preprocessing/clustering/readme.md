
### Clustering of Patch Embeddings
<br>

Check **sample script** used for clustering patch embeddings in the TCGA_LUAD dataset: ```scripts\run_cluster_patch_embeddings_tcga_luad.sh```  

Check **sample script** used for predicting clusters of the patches in the DHMC_LUAD dataset based on TCGA_LUAD patch clustering ```scripts\run_cluster_predict_patch_embeddings_dhmc_luad.sh```  
<br>

- **```cluster_patches.py```**  

	Cluster patch embeddings or predict patch clusters using previously computed cluster centroids.  
	
	Input parameters:  
	- **```root_tile_embeddings_dir```:** Input directory containing the WSI patch embeddings directory.  
	- **```root_out_dir```:** Output directory.  
	- **```cluster_center_filepath```:** If using an existing clustering, specify the clustering centroids numpy filepath. Default is None indicating to perform clustering from scratch.  
	- **```n_clusters_preference```:** The number of clusters. Default is 16.  
	- **```clustering_method_name```:** Clustering method name. Possible values: kmeans and afp. Default is kmeans.  

	Sample clustering is available at: ```datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16```  



<br>


- **```comp_maps_n_stats.py```**
	
	1. Compute foreground mask (```<slide_name>_fg_mask.png```), background mask (```<slide_name>_bg_mask.png```), and cluster IDs map (```<slide_name>_cluster_id_map.npy```) from the clustered patches.   \
	Each patch is represented by one pixel in the generated maps.  \
	The maps are output to the directory ```<root_out_dir>/clustering_maps```.
	2. Compute statistics of the clusters in each slide (```<root_out_dir>/clustering_stats/cluster_stats.csv```).

	Input parameters:  
	- **```clustering_assignment_dir```:** Output directory from the previous clustering that contains the predicted patch clusters.  
	- **```n_clusters```:** The number of clusters. Default is 16.  
	- **```root_out_dir```:** Output directory.  
	- **```wsi_dim_filepath```:** Input csv file contains meta data for the slide. It has the columns: ```slide_name```, ```slide_width```,	```slide_height```,	```slide_mag```, ```slide_pw```, where ```slide_pw``` is the tile size at the highest magnification. 
	- **```ignore_cluster_ids```:** A list of cluster IDs to exclude. This is useful for excluding artifacts and other irrelevant or background regions in the slide. Default is None.

	Sample maps and stats files are available under: ```datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16```  

<br>

- **```clustering_vis.py```**
	
	Visualize the cluster maps. Each cluster is represented by a different color in a 50x50 square region. The color dictionary currently support up 24 clusters.   \

	Input parameters:  
	- **```clustering_assignment_dir```:** Output directory from the previous clustering that contains the predicted patch clusters.  
	- **```n_clusters```:** The number of clusters. Default is 16.  
	- **```root_out_dir```:** Output directory.  
	- **```wsi_dim_filepath```:** Input csv file contains meta data for the slide. It has the columns: ```slide_name```, ```slide_width```,	```slide_height```,	```slide_mag```, ```slide_pw```, where ```slide_pw``` is the tile size at the highest magnification. 
	- **```ignore_cluster_ids```:** A list of cluster IDs to exclude. Default is None.

	Sample visualization maps are available under: ```datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16```  

