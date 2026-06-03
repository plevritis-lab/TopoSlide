#!/bin/bash
SCR_PATH=`dirname -- $BASH_SOURCE` 
cd $SCR_PATH;
cd ../preprocessing/cluster_topology
pwd;
nohup python comp_cluster_topology.py  \
		--clustering_assignment_dir "/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans" \
		--n_clusters 16 \
		--wsi_dim_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv" \
		--root_out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp" \
		--visualize \
		>> ../../log/log_run_comp_topo_tcga_luad_1.txt &
    
nohup python comp_cluster_topology.py  \
		--clustering_assignment_dir "/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans" \
		--n_clusters 16 \
		--wsi_dim_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv" \
		--root_out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp" \
		--visualize \
		>> ../../log/log_run_comp_topo_tcga_luad_2.txt &

nohup python comp_cluster_topology.py  \
		--clustering_assignment_dir "/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans" \
		--n_clusters 16 \
		--wsi_dim_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv" \
		--root_out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp" \
		--visualize \
		>> ../../log/log_run_comp_topo_tcga_luad_3.txt &

nohup python comp_cluster_topology.py  \
		--clustering_assignment_dir "/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans" \
		--n_clusters 16 \
		--wsi_dim_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv" \
		--root_out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp" \
		--visualize \
		>> ../../log/log_run_comp_topo_tcga_luad_4.txt &

	
wait

nohup python save_to_hdf_cluster_topology.py  \
		--root_input_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp" \
		--out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo" \
		>> ../../log/log_run_save_to_hdf_tcga_luad_1.txt &
    
nohup python save_to_hdf_cluster_topology.py  \
		--root_input_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp" \
		--out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo" \
		>> ../../log/log_run_save_to_hdf_tcga_luad_2.txt &

nohup python save_to_hdf_cluster_topology.py  \
		--root_input_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp" \
		--out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo" \
		>> ../../log/log_run_save_to_hdf_tcga_luad_3.txt &

nohup python save_to_hdf_cluster_topology.py  \
		--root_input_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp" \
		--out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo" \
		>> ../../log/log_run_save_to_hdf_tcga_luad_4.txt &
wait


nohup python comp_topo_hist_norm_factor.py  \
		--root_input_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo" \
		--out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo/stats" \
		--n_clusters 16 \
		--wsi_dim_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv" \
		>> ../../log/log_run_comp_topo_hist_norm_factor.txt &

wait