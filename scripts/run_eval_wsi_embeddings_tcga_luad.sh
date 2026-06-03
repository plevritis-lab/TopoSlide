#!/bin/bash
SCR_PATH=`dirname -- $BASH_SOURCE` 
cd $SCR_PATH;
cd ../eval_wsi_embeddings
pwd;
nohup python run_toposlide_eval.py  \
		--patch_embedding_dir "/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/patches_20x_512_conch_tile_embedding" \
		--slide_root_output_dir "/oak/stanford/groups/plevriti/shahira/datasets/tcga_luad/tcga_luad_toposlide_wsi_20x_512" \
		--checkpoint_path "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/trained_models/toposlide_tcga_luad.pt" \
		>> ../log/log_run_toposlide_eval_tcga_luad.txt &

wait
