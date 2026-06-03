
### Compute Topology of Clusters

Compute the 0D and 1D topology formed by patches in each cluster and save to HDF5 files.

- Check sample **script** used for computing topology of clusters in the TCGA_LUAD dataset: ```scripts\run_comp_topo_tcga_luad.sh```  


1. ```comp_cluster_topology.py```  
	
-	Input parameters are:  \
	**clustering_assignment_dir:** Input directory containing the cluster assignments of the WSI patch embeddings.  \
	**n_clusters:** The number of clusters. Default is 16.  \
	**wsi_dim_filepath:** Input csv file contains meta data for the slide. It has the columns: ```slide_name```, ```slide_width```,	```slide_height```,	```slide_mag```, ```slide_pw```, where ```slide_pw``` is the tile size at the highest magnification. \
	**ignore_cluster_ids:** A list of cluster IDs to exclude their patches when computing the WSI embedding. This is useful for excluding artifacts and other irrelevant or background regions in the slide. Default is None.\
	**root_out_dir:** Output directory.  \
	**visualize:** If present, then save topology visualization. Default is False.  


2. ```save_to_hdf_cluster_topology.py```  
	
-	Input parameters are:  \
	**root_input_dir:** Input directory containing the clusters topology within each WSI.  \
	**out_dir:** Output directory  

3. ```comp_topo_hist_norm_factor.py```  
-	Computes the topology histogram normalization array for 0D and 1D topology using statistics from the data.
-	Input parameters are:  \
	**root_input_dir:** Input directory containing the clusters topology within each WSI in HDF5 format.  \
	**out_dir:** Output directory.  \
	**num_clusters:** The number of clusters. Default is 16.  \
	**wsi_dim_filepath:** Input csv file contains meta data for the slide. It has the columns: ```slide_name```, ```slide_width```,	```slide_height```,	```slide_mag```, ```slide_pw```, where ```slide_pw``` is the tile size at the highest magnification. \
