#!/bin/bash
SCR_PATH=`dirname -- $BASH_SOURCE` 
cd $SCR_PATH;
cd ../preprocessing/extract_meta_data
pwd;
nohup python extract_wsi_dim_dhmc.py  \
		--svs_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/svs" \
		--out_dir "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad" \
		--dataset_name "dhmc_luad" \
		--svs_meta_filepath "/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/dhmc_luad/MetaData_Release_1.0.csv" \
		>> ../../log/log_run_extract_wsi_dim_dhmc_luad.txt &
wait
