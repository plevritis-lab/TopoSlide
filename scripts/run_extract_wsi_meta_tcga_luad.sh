#!/bin/bash
SCR_PATH=`dirname -- $BASH_SOURCE` 
cd $SCR_PATH;
cd ../preprocessing/extract_meta_data
pwd;
nohup python extract_wsi_dim.py  \
		--svs_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/svs" \
		--out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad" \
		--dataset_name "tcga_luad" \
		>> ../../log/log_run_extract_wsi_dim_tcga_luad.txt &

wait
