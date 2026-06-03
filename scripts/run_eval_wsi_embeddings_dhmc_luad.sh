#!/bin/bash
SCR_PATH=`dirname -- $BASH_SOURCE` 
cd $SCR_PATH;
cd ../eval_wsi_embeddings
pwd;
nohup python run_toposlide_eval_w_exclude.py  \
		--patch_embedding_dir "/oak/stanford/groups/plevriti/shahira/datasets/dhmc_luad/patches_20x_512_conch_tile_embedding" \
		--slide_root_output_dir "/oak/stanford/groups/plevriti/shahira/datasets/dhmc_luad/dhmc_luad_toposlide_wsi_20x_512" \
		--checkpoint_path "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/trained_models/toposlide_tcga_luad.pt" \
		--clustering_dir "/oak/stanford/groups/plevriti/shahira/datasets/dhmc_luad/patches_20x_512_conch_tile_embedding_tcga_luad_kmeans_n16/prediction_kmeans" \
		--wsi_dim_filepath "/oak/stanford/groups/plevriti/shahira/dhmc_luad_wsi_dim_p512_m20.csv" \
		--ignore_cluster_ids 1 8 0 15 \
		--n_clusters 16 \
		>> ../log/log_run_toposlide_eval_dhmc_luad.txt &

wait
