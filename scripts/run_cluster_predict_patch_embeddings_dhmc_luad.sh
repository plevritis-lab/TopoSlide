#!/bin/bash
SCR_PATH=`dirname -- $BASH_SOURCE` 
cd $SCR_PATH;
cd ../preprocessing/clustering
pwd;
nohup python cluster_patches.py  \
		--root_tile_embeddings_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/patches_20x_512_conch_tile_embedding" \
		--n_clusters_preference 16 \
		--root_out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/patches_20x_512_conch_tile_embedding_tcga_luad_kmeans_n16" \
		--cluster_center_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/kmeans_clustering_conch512_n16_centers.npy"  \
		>> ../../log/log_run_cluster_predict_dhmc_luad.txt &
    
wait

    
nohup python comp_maps_n_stats.py  \
		--clustering_assignment_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/patches_20x_512_conch_tile_embedding_tcga_luad_kmeans_n16/prediction_kmeans" \
		--n_clusters 16 \
		--root_out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/patches_20x_512_conch_tile_embedding_tcga_luad_kmeans_n16" \
		--ignore_cluster_ids 1 8 0 15 \
		--wsi_dim_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/dhmc_luad2_wsi_dim_p512_m20.csv"  \
		>> ../../log/log_run_cluster_predict_dhmc_luad_comp_maps.txt &
		
wait

nohup python clustering_vis.py  \
		--clustering_assignment_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/patches_20x_512_conch_tile_embedding_tcga_luad_kmeans_n16/prediction_kmeans" \
		--n_clusters 16 \
		--root_out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/patches_20x_512_conch_tile_embedding_tcga_luad_kmeans_n16/prediction_kmeans_vis" \
		--wsi_dim_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/dhmc_luad2_wsi_dim_p512_m20.csv"  \
		>> ../../log/log_run_cluster_predict_dhmc_luad_vis.txt &
		
wait
