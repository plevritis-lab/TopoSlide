import sys
import os
import glob
import math
import gc
import logging
import argparse

import numpy as np
from skimage import io
from skimage.measure import label
from sklearn import metrics
from scipy.spatial import ConvexHull
from scipy import ndimage
import pandas as pd
import cv2

from topology_image_utils import read_coord_scaled_csv, compute_distance_transform, vis_save_topo, compute_topo, plot_pers_diag, compute_dtm_k_frac, compute_dtm_k_fixed


def configure_logger(log_filepath):    
    logging.basicConfig(level=logging.ERROR, filename=log_filepath, filemode='a+', format='%(name)s - (asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    # logging.basicConfig(level=logging.DEBUG, filename=log_filepath, filemode='a+', format='%(name)s - (asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    # Logging usage examples:
    # logging.debug(f"In load_slide({slide_filepath}, {self.slide_id})")
    # logging.info(f"Slide {slide_filepath} loaded successfully.")
    # logging.error(f"Slide {slide_filepath} load failed.")


def box_kernel(size):
  k = np.ones((size,size),np.float32)/(size**2)
  return k

def ones_kernel(size):
  k = np.ones((size,size),np.float32)
  return k

def save_topo_to_csv(bcp_coord, dcp_coord, bcp_val, dcp_val, pers_val, padding_width, scale, out_filepath):
    bcp_coord = (bcp_coord - padding_width)/scale
    dcp_coord = (dcp_coord - padding_width)/scale

    df = pd.DataFrame()
    df['bcp_y'] = bcp_coord[:,0]
    df['bcp_x'] = bcp_coord[:,1]
    df['dcp_y'] = dcp_coord[:,0]
    df['dcp_x'] = dcp_coord[:,1]
    df['bcp_val'] = bcp_val
    df['dcp_val'] = dcp_val
    df['pers'] = pers_val

    df.to_csv(out_filepath, index=False)
    return

def vis_cp_pair_overlaid(base_img, bcp, dcp, pers, out_filepath):

    # topo_vis_cp_pair = np.zeros((height, width))
    #topo_vis_bcp[(bcp[:,0], bcp[:,1])] = 1
    #topo_vis_dcp[(dcp[:,0], dcp[:,1])] = 1

    base_img = base_img[:,:,::-1] # Using OpenCV here. Neet to convert to BGR.
    for i in range(len(bcp[:,0])):
        # base_img = cv2.line(base_img[:,:,::-1], bcp[i][::-1], dcp[i][::-1], color_dict["cp_pair"], int(min_line_thickness + pers[i]/pers.max() * (max_line_thickness - min_line_thickness)))
        try:
            base_img = cv2.line(base_img, bcp[i][::-1], dcp[i][::-1], color_dict["cp_pair"], int(min_line_thickness + pers[i]/pers.max() * (max_line_thickness - min_line_thickness)))
        except Exception as e:
            print('exception *******')
            print(e)
            print('bcp[i][::-1]', bcp[i][::-1])
            print('dcp[i][::-1]', dcp[i][::-1])
            print('pers[i]',pers[i])
            print('pers.max()',pers.max())
            raise e
        # rr, cc = line(bcp[i,0], bcp[i,1], dcp[i,0], dcp[i,1])
        # base_img[rr, cc] = 1
    if(out_filepath is not None):
        io.imsave(out_filepath, (base_img).astype(np.uint8), check_contrast=False)

    return

def vis_cp_pairs(height, width, bcp, dcp, pers):

    im = np.zeros((height, width, 3))
    for i in range(len(bcp[:,0])):
        try:
            im = cv2.line(im , bcp[i][::-1], dcp[i][::-1], color_dict["cp_pair"], int(min_line_thickness + pers[i]/pers.max() * (max_line_thickness - min_line_thickness)))
        except Exception as e:
            print('exception *******')
            print(e)
            print('bcp[i][::-1]', bcp[i][::-1])
            print('dcp[i][::-1]', dcp[i][::-1])
            print('pers[i]',pers[i])
            print('pers.max()',pers.max())
            raise e

    return im

def hex_to_rgb(hex_color_list):
    rgb_color_array = np.zeros((len(hex_color_list), 3))
    for indx, hex_color_code in enumerate(hex_color_list):
        hex_color_code = hex_color_code.lstrip('#')  # Remove '#' if present

        if len(hex_color_code) != 6:
            return None  # Invalid hex code length

        try:
            r = int(hex_color_code[0:2], 16)
            g = int(hex_color_code[2:4], 16)
            b = int(hex_color_code[4:6], 16)
            rgb_color_array[indx] = [r, g, b]
        except ValueError:
            return None  # Invalid hexadecimal characters
    return rgb_color_array


def compute_topo_cc(group_name, group_indx, im_cc, out_dir, foreground_mask_padded, background_mask_padded, new_height, new_width):

    if(dtm_k <= 1):
        im_dist = ndimage.distance_transform_edt(im_cc)
    else:
        (ref_points_y, ref_points_x) = np.where(im_cc==0)
        im_dist = compute_dtm_k_fixed(dtm_k, new_height, new_width, ref_points_y, ref_points_x, out_dir if visualize else None, pnt_cloud_name if visualize else None )


    # pad im_dist
    tmp_map = np.ones((im_dist.shape[0]+padding_width*2, im_dist.shape[1]+padding_width*2))
    tmp_map[padding_width:-padding_width, padding_width:-padding_width] = im_dist
    im_dist = tmp_map

    # set bg to -inf
    im_dist_vals = im_dist.copy()
    im_dist[background_mask_padded>0] = math.inf

    pdl, bcp, dcp, pers = compute_topo(im_dist, pad_value=math.inf)
    if(len(pers)>1):
        # exclude topo comp outside tissue
        # logging.info(f"Exclude topo comp outside tissue")
        is_bcp_fg = foreground_mask_padded[(bcp[:,0], bcp[:,1])]
        is_dcp_fg = foreground_mask_padded[(dcp[:,0], dcp[:,1])]
        pdl = pdl[np.logical_and(is_bcp_fg,is_dcp_fg) ]
        bcp = bcp[np.logical_and(is_bcp_fg,is_dcp_fg) ]
        dcp = dcp[np.logical_and(is_bcp_fg,is_dcp_fg) ]
        pers = pers[np.logical_and(is_bcp_fg,is_dcp_fg) ]

        if(len(pers)>=1):
            bcp_val = im_dist[(bcp[:,0], bcp[:,1])]
            dcp_val = im_dist[(dcp[:,0], dcp[:,1])]
            pers_val = np.abs(bcp_val - dcp_val)

            overlay_img = foreground_mask_padded
            overlay_img=overlay_img[:,:,np.newaxis] 
            overlay_img=np.concatenate((overlay_img,overlay_img,overlay_img),2) * color_dict["fg"] 
            overlay_img[(tiles_coord_y_scaled[tiles_cluster_id==ci], tiles_coord_x_scaled[tiles_cluster_id==ci])] = cluster_color_arr[ci]
            out_filepath = os.path.join(out_dir, pnt_cloud_name + '_pnts_cc_exclude_bg' + "_topo_cp_pair_overlay.png")
            cp_pairs_im = vis_cp_pairs(new_height+padding_width*2, new_width+padding_width*2, bcp, dcp, pers_val)
            overlay_img[cp_pairs_im>0] = cp_pairs_im[cp_pairs_im>0]
            io.imsave(out_filepath, overlay_img.astype(np.uint8), check_contrast=False)


            plot_pers_diag(dcp_val, bcp_val, out_dir, pnt_cloud_name + '_pnts_cc_exclude_bg', max_val=bcp_val.max())

            out_filepath = os.path.join(out_dir, f"{pnt_cloud_name}_pnts_cc_exclude_bg_topo_cps.csv")
            save_topo_to_csv(bcp, dcp, bcp_val, dcp_val, pers_val, 0, 1, out_filepath)
            out_filepath = os.path.join(out_dir, f"{pnt_cloud_name}_pnts_cc_exclude_bg_topo_cps_original_coord.csv")
            save_topo_to_csv(bcp, dcp, bcp_val, dcp_val, pers_val, padding_width, scale, out_filepath)
    return


def compute_topo_holes(group_name, group_indx, im_cc, out_dir, foreground_mask_padded, background_mask_padded, new_height, new_width):

    if(im_cc.sum()<min_pnts):
        return

    hull_im = np.zeros((new_height, new_width))
    fg_lbl = label(foreground_mask_padded)
    for comp_i in range(1, fg_lbl.max()+1):
        fg_comp = (fg_lbl==comp_i)
        if(fg_comp.sum()<min_pnts):
            continue
        im_cc_comp = np.logical_and(im_cc, fg_comp[padding_width:-padding_width,padding_width:-padding_width])
        if(im_cc_comp.sum()<min_pnts):
            continue
        points_y, points_x = np.where(im_cc_comp>0)
        points_2d =np.stack((points_y,points_x), axis=1)
        try:
            hull = ConvexHull(points_2d)
        except:
            continue

        hull_im_comp = np.zeros((new_height, new_width))
        for simplex in hull.simplices:
            hull_im_comp = cv2.line(hull_im_comp , points_2d[simplex[0]][::-1],points_2d[simplex[1]][::-1], 255, int(min_line_thickness))

        hull_points = points_2d[hull.vertices].astype(np.int32)
        hull_contour = hull_points.reshape((-1, 1, 2))
        cv2.fillPoly(hull_im_comp, pts=[hull_contour[:,:,::-1]], color=255)
        hull_im[hull_im_comp>0] = 1

    if(hull_im.sum()<min_pnts):
        return
    io.imsave(os.path.join(out_dir, f"{pnt_cloud_name}_hull_filled.png"), (hull_im*255).astype(np.uint8), check_contrast=False)

    hull_im[background_mask>0] = 0
    if(hull_im.sum()<min_pnts):
        return
    io.imsave( os.path.join(out_dir, f"{pnt_cloud_name}_hull_filled_fg.png"), (hull_im*255).astype(np.uint8), check_contrast=False)

    foreground_mask_ci = hull_im
    foreground_mask_ci[foreground_mask_ci>0] = 1
    background_mask_ci = np.zeros(foreground_mask_ci.shape)
    background_mask_ci[foreground_mask_ci<=0] = 1

    tmp_map = np.zeros((foreground_mask_ci.shape[0]+padding_width*2, foreground_mask_ci.shape[1]+padding_width*2))
    tmp_map[padding_width:-padding_width, padding_width:-padding_width] = foreground_mask_ci
    foreground_mask_ci_padded = tmp_map

    tmp_map = np.ones((background_mask_ci.shape[0]+padding_width*2, background_mask_ci.shape[1]+padding_width*2))
    tmp_map[padding_width:-padding_width, padding_width:-padding_width] = background_mask_ci
    background_mask_ci_padded = tmp_map

    (ref_points_y, ref_points_x) = np.where(im_cc==1)
    if(dtm_k <= 1):
        im_dist = compute_distance_transform(new_height, new_width, ref_points_y, ref_points_x, out_dir if visualize else None, pnt_cloud_name if visualize else None )
    else:
        im_dist = compute_dtm_k_fixed(dtm_k, new_height, new_width, ref_points_y, ref_points_x, out_dir if visualize else None, pnt_cloud_name if visualize else None )


    # pad im_dist
    tmp_map = np.ones((im_dist.shape[0]+padding_width*2, im_dist.shape[1]+padding_width*2))
    tmp_map[padding_width:-padding_width, padding_width:-padding_width] = im_dist
    im_dist = tmp_map

    im_dist_vals = im_dist.copy()
    im_dist[background_mask_padded>0] = math.inf
    pdl, bcp, dcp, pers = compute_topo(im_dist, pad_value=-math.inf)
    if(len(pers)>1):
      
        k = np.ones((3, 3), np.uint8)  # Define 3x3 kernel
        foreground_mask_ci_padded_eroded = cv2.erode(foreground_mask_ci_padded.copy(), k, 1)    
        foreground_mask_ci_padded_dilated = cv2.dilate(foreground_mask_ci_padded.copy(), k, 1)    

        # exclude topo comp outside tissue
        # logging.info(f"Exclude topo comp outside tissue")
        is_bcp_fg = foreground_mask_padded_dilated[(bcp[:,0], bcp[:,1])]
        is_dcp_fg = foreground_mask_padded_dilated[(dcp[:,0], dcp[:,1])]
        pdl = pdl[np.logical_and(is_bcp_fg,is_dcp_fg) ]
        bcp = bcp[np.logical_and(is_bcp_fg,is_dcp_fg) ]
        dcp = dcp[np.logical_and(is_bcp_fg,is_dcp_fg) ]
        pers = pers[np.logical_and(is_bcp_fg,is_dcp_fg) ]
        is_bcp_fg = foreground_mask_padded_eroded[(bcp[:,0], bcp[:,1])]
        is_dcp_fg = foreground_mask_padded_eroded[(dcp[:,0], dcp[:,1])]
        pdl = pdl[np.logical_or(is_bcp_fg,is_dcp_fg) ]
        bcp = bcp[np.logical_or(is_bcp_fg,is_dcp_fg) ]
        dcp = dcp[np.logical_or(is_bcp_fg,is_dcp_fg) ]
        pers = pers[np.logical_or(is_bcp_fg,is_dcp_fg) ]

        if(len(pers)>1):
            bcp_val = im_dist_vals[(bcp[:,0], bcp[:,1])]
            dcp_val = im_dist_vals[(dcp[:,0], dcp[:,1])]
            pers_val = np.abs(bcp_val - dcp_val)

            overlay_img = foreground_mask_padded
            overlay_img=overlay_img[:,:,np.newaxis]
            overlay_img=np.concatenate((overlay_img,overlay_img,overlay_img),2) * color_dict["fg"] 
            overlay_img[(tiles_coord_y_scaled[tiles_cluster_id==ci], tiles_coord_x_scaled[tiles_cluster_id==ci])] = cluster_color_arr[ci]
            out_filepath = os.path.join(out_dir, pnt_cloud_name + '_pnts_holes_exclude_bg' + "_topo_cp_pair_overlay.png")
            cp_pairs_im = vis_cp_pairs(new_height+padding_width*2, new_width+padding_width*2, bcp, dcp, pers_val)
            overlay_img[cp_pairs_im>0] = cp_pairs_im[cp_pairs_im>0]
            io.imsave(out_filepath, overlay_img.astype(np.uint8), check_contrast=False)

            plot_pers_diag(dcp_val, bcp_val, out_dir, pnt_cloud_name + '_pnts_holes_exclude_bg', max_val=bcp_val.max())

            out_filepath = os.path.join(out_dir, f"{pnt_cloud_name}_pnts_holes_exclude_bg_topo_cps.csv")
            save_topo_to_csv(bcp, dcp, bcp_val, dcp_val, pers_val, 0, 1, out_filepath)
            out_filepath = os.path.join(out_dir, f"{pnt_cloud_name}_pnts_holes_exclude_bg_topo_cps_original_coord.csv")
            save_topo_to_csv(bcp, dcp, bcp_val, dcp_val, pers_val, padding_width, scale, out_filepath)


            # exclude topo comp outside tissue
            # logging.info(f"Exclude topo comp outside convex hull fg")
            is_bcp_fg = foreground_mask_ci_padded_dilated[(bcp[:,0], bcp[:,1])]
            is_dcp_fg = foreground_mask_ci_padded_dilated[(dcp[:,0], dcp[:,1])]
            pdl = pdl[np.logical_or(is_bcp_fg,is_dcp_fg) ]
            bcp = bcp[np.logical_or(is_bcp_fg,is_dcp_fg) ]
            dcp = dcp[np.logical_or(is_bcp_fg,is_dcp_fg) ]
            pers = pers[np.logical_or(is_bcp_fg,is_dcp_fg) ]

            if(len(pers)>1):
                bcp_val = im_dist_vals[(bcp[:,0], bcp[:,1])]
                dcp_val = im_dist_vals[(dcp[:,0], dcp[:,1])]
                pers_val = np.abs(bcp_val - dcp_val)

                overlay_img = foreground_mask_ci_padded
                overlay_img=overlay_img[:,:,np.newaxis]
                overlay_img=np.concatenate((overlay_img,overlay_img,overlay_img),2) * color_dict["fg"] 
                overlay_img[(tiles_coord_y_scaled[tiles_cluster_id==ci], tiles_coord_x_scaled[tiles_cluster_id==ci])] = cluster_color_arr[ci]
                out_filepath = os.path.join(out_dir, pnt_cloud_name + '_pnts_holes_exclude_bg_filter_hull' + "_topo_cp_pair_overlay.png")
                cp_pairs_im = vis_cp_pairs(new_height+padding_width*2, new_width+padding_width*2, bcp, dcp, pers_val)
                overlay_img[cp_pairs_im>0] = cp_pairs_im[cp_pairs_im>0]
                io.imsave(out_filepath, overlay_img.astype(np.uint8), check_contrast=False)

                plot_pers_diag(dcp_val, bcp_val, out_dir, pnt_cloud_name + '_pnts_holes_exclude_bg_filter_hull', max_val=bcp_val.max())

                out_filepath = os.path.join(out_dir, f"{pnt_cloud_name}_pnts_holes_exclude_bg_filter_hull_topo_cps.csv")
                save_topo_to_csv(bcp, dcp, bcp_val, dcp_val, pers_val, 0, 1, out_filepath)
                out_filepath = os.path.join(out_dir, f"{pnt_cloud_name}_pnts_holes_exclude_bg_filter_hull_topo_cps_original_coord.csv")
                save_topo_to_csv(bcp, dcp, bcp_val, dcp_val, pers_val, padding_width, scale, out_filepath)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Compute topology formed by patch clusters")    

    parser.add_argument("--clustering_assignment_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_kmeans_n16/prediction_kmeans")
    parser.add_argument("--n_clusters", type=int, default=16)
    parser.add_argument("--wsi_dim_filepath", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/tcga_luad_wsi_dim_p512_m20.csv")
    parser.add_argument("--ignore_cluster_ids", type=int, nargs="+", default=None) # [1,8,0,15]
    parser.add_argument("--root_out_dir", type=str, default="/oak/stanford/groups/plevriti/shahira/code/TopoSlide/datasets/tcga_luad/patches_20x_512_conch_tile_embedding_clustering_topo_tmp")
    parser.add_argument("--visualize", action='store_true', default=False)

    args = parser.parse_args()

    clustering_assignment_dir = args.clustering_assignment_dir
    n_clusters = args.n_clusters
    wsi_dim_filepath = args.wsi_dim_filepath
    root_out_dir = args.root_out_dir
    visualize = args.visualize

    ignore_clusters_arr = None
    if(args.ignore_cluster_ids is not None and len(args.ignore_cluster_ids)>0):
        ignore_clusters_arr = np.array(list(set(args.ignore_cluster_ids)))

    padding_width = 1
    min_pnts=5
    dtm_k = 5 # here number of points in point cloud is not related to the density but rather the area. Therefore it is better to fix the k rather than compute based on fraction of points in point cloud
    tile_vis_size = 1

    groups_dict = {} # {<group_name>:[<group_indx>, list of cluster ids]}
    for i in range(n_clusters):
        groups_dict[f"c{i}"]=[i,[i]]
    n_groups = n_clusters

    hist_buckets_arr = [2, 5,10,20,30,40,50, math.inf]

    color_dict = {"fg":[128, 128, 128], "cp_pair":[255, 255, 255]} 
    max_line_thickness = 1
    min_line_thickness = 1
    colors16 = [
        '#f2c0a2',
        '#e98472',
        '#d82323',
        '#98183c',
        '#1fcb23',
        '#126d30',
        '#26dddd',
        '#1867a0',
        '#934226',
        '#6c251e',
        '#f7e26c',
        '#edb329',
        '#e76d14',
        '#f2f2f9',
        '#6a5fa0',
        '#161423',
        ]

    done_filename = "topo_done.txt"

    if(not os.path.exists(root_out_dir)):
        os.makedirs(root_out_dir, exist_ok=True)
    

    log_filepath = os.path.join(root_out_dir, 'patch_topo_analysis_log.txt')
    configure_logger(log_filepath)
    logging.debug(f"in main")

    cluster_color_arr = hex_to_rgb(colors16)

    wsi_dim_df = pd.read_csv(wsi_dim_filepath)
    wsi_dim_arr = wsi_dim_df.to_numpy()


    wi = 0
    # for indx in range(wsi_dim_arr.shape[0]):
    for indx in np.random.permutation(wsi_dim_arr.shape[0]):
        

        slide_name = wsi_dim_arr[indx,0]

        print(indx, slide_name, end='\r')

        out_dir = os.path.join(root_out_dir, slide_name)
        if(not os.path.exists(out_dir)):
            os.mkdir(out_dir)
        done_filepath = os.path.join(out_dir, done_filename)
        if(os.path.exists(done_filepath)):
            print('Done')
            continue

        logging.info(f"slide_name={slide_name}")
        tile_size = -1
        if(wsi_dim_arr.shape[-1] == 3):
            _, width, height = wsi_dim_arr[wsi_dim_arr[:,0]==slide_name][0]
        else:
            _, width, height, mag, tile_size  = wsi_dim_arr[wsi_dim_arr[:,0]==slide_name][0]
        logging.info(f"height={height}")
        logging.info(f"width={width}")

        files = glob.glob(os.path.join(clustering_assignment_dir, f"{slide_name}*.csv"))
        if(files is None or len(files)==0):
            print('No Clustering Info')
            continue
        clustering_filepath = files[0]       
        clustering_df = pd.read_csv(clustering_filepath)

        tiles_coord_x = clustering_df["coord_x"].to_numpy()
        tiles_coord_y = clustering_df["coord_y"].to_numpy()
        tiles_cluster_id = clustering_df["cluster_id"].to_numpy()
        tiles_cluster_dist = clustering_df["cluster_dist"].to_numpy()
        if(tile_size <= 0):
            found_tile_size = False
            ti = 0
            while(ti+1<len(tiles_coord_x)):
                x1 = tiles_coord_x[ti]
                x2 = tiles_coord_x[ti+1]
                y1 = tiles_coord_y[ti]
                y2 = tiles_coord_y[ti+1]
                tile_size = min(abs(x2-x1), abs(y2-y1))   
                if(tile_size > 0 and tile_size < 600):
                    found_tile_size = True
                    break
                tile_size = max(abs(x2-x1), abs(y2-y1))   
                if(tile_size < 600):
                    found_tile_size = True
                    break
                ti += 1
            if(not found_tile_size):
                ti = 0
                while(ti+1<len(tiles_coord_x)):
                    x1 = tiles_coord_x[ti]
                    x2 = tiles_coord_x[ti+1]
                    y1 = tiles_coord_y[ti]
                    y2 = tiles_coord_y[ti+1]
                    tile_size = min(abs(x2-x1), abs(y2-y1))   
                    if(tile_size > 0 and tile_size < 1100):
                        found_tile_size = True
                        break
                    tile_size = max(abs(x2-x1), abs(y2-y1))   
                    if(tile_size < 1100):
                        found_tile_size = True
                        break
                    ti += 1
            if(not found_tile_size):
                ti = 0
                while(ti+1<len(tiles_coord_x)):
                    x1 = tiles_coord_x[ti]
                    x2 = tiles_coord_x[ti+1]
                    y1 = tiles_coord_y[ti]
                    y2 = tiles_coord_y[ti+1]
                    tile_size = min(abs(x2-x1), abs(y2-y1))   
                    if(tile_size > 0 and tile_size < 1600):
                        found_tile_size = True
                        break
                    tile_size = max(abs(x2-x1), abs(y2-y1))   
                    if(tile_size < 1600):
                        found_tile_size = True
                        break
                    ti += 1
            if(not found_tile_size):
                ti = 0
                while(ti+1<len(tiles_coord_x)):
                    x1 = tiles_coord_x[ti]
                    x2 = tiles_coord_x[ti+1]
                    y1 = tiles_coord_y[ti]
                    y2 = tiles_coord_y[ti+1]
                    tile_size = min(abs(x2-x1), abs(y2-y1))   
                    if(tile_size > 0 and tile_size < 2100):
                        found_tile_size = True
                        break
                    tile_size = max(abs(x2-x1), abs(y2-y1))   
                    if(tile_size < 2100):
                        found_tile_size = True
                        break
                    ti += 1
        scale = tile_vis_size/tile_size
        new_height = int(height*scale+1)
        new_width = int(width*scale+1)

        if(ignore_clusters_arr is not None):
            clustering_df = clustering_df[~clustering_df['cluster_id'].isin(ignore_clusters_arr)] 

            tiles_coord_x = clustering_df["coord_x"].to_numpy()
            tiles_coord_y = clustering_df["coord_y"].to_numpy()
            tiles_cluster_id = clustering_df["cluster_id"].to_numpy()
            tiles_cluster_dist = clustering_df["cluster_dist"].to_numpy()

        tiles_coord_x_scaled = np.round(tiles_coord_x*scale).astype(int)
        tiles_coord_y_scaled = np.round(tiles_coord_y*scale).astype(int)
        tiles_coord_x_scaled[tiles_coord_x_scaled>=new_width ] = new_width-1
        tiles_coord_y_scaled[tiles_coord_y_scaled>=new_height ] = new_height-1
        print('tiles_coord_x_scaled.max()', tiles_coord_x_scaled.max())
        print('tiles_coord_y_scaled.max()', tiles_coord_y_scaled.max())

        cluster_id_map = np.ones((new_height, new_width))*-1
        for ti in range(len(tiles_coord_x_scaled)):
            cluster_id_map[tiles_coord_y_scaled[ti]:min(new_height, tiles_coord_y_scaled[ti]+tile_vis_size), tiles_coord_x_scaled[ti]:min(new_width, tiles_coord_x_scaled[ti]+tile_vis_size)] = tiles_cluster_id[ti]
        foreground_mask = cluster_id_map>=0
        background_mask = cluster_id_map<0        
        io.imsave( os.path.join(out_dir, f"fg_mask.png"), (foreground_mask*255).astype(np.uint8), check_contrast=False)
        io.imsave( os.path.join(out_dir, f"bg_mask.png"), (background_mask*255).astype(np.uint8), check_contrast=False)
        cluster_id_map.astype(int).dump(os.path.join(out_dir, f"cluster_id_map.npy"))

        distances_arr = clustering_df[[f"dist_{ci}" for ci in range(n_clusters)]].to_numpy()
        
        patch_denom = distances_arr.sum(axis=-1)

        masks = np.zeros((n_clusters, new_height, new_width))
        point_maps = np.zeros((n_clusters, new_height, new_width))
        for ci in range(n_clusters):            
            tiles_cluster_dist_ci_norm = clustering_df[f"dist_{ci}"].to_numpy() 
            tiles_cluster_dist_ci_norm = 1-(tiles_cluster_dist_ci_norm  / patch_denom) # sdf2
            fpr, tpr, thresholds = metrics.roc_curve(tiles_cluster_id==ci,tiles_cluster_dist_ci_norm)
            thresh = thresholds[np.argmax(tpr - fpr)]
            masks[(ci, tiles_coord_y_scaled[tiles_cluster_dist_ci_norm>=thresh], tiles_coord_x_scaled[tiles_cluster_dist_ci_norm>=thresh])]=1
            masks[(ci, tiles_coord_y_scaled[tiles_cluster_id==ci], tiles_coord_x_scaled[tiles_cluster_id==ci])]=1
            io.imsave( os.path.join(out_dir, f"mask_c{ci}.png"), (masks[ci]*255).astype(np.uint8), check_contrast=False)
            point_maps[(ci, tiles_coord_y_scaled[tiles_cluster_id==ci], tiles_coord_x_scaled[tiles_cluster_id==ci])]=1
            io.imsave( os.path.join(out_dir, f"pnts_c{ci}.png"), (point_maps[ci]*255).astype(np.uint8), check_contrast=False)
        masks_sum = masks.sum(axis=0)
        io.imsave( os.path.join(out_dir, f"masks_sum.png"), ((masks_sum/n_clusters)*255).astype(np.uint8), check_contrast=False)

        masks_groups = np.zeros((n_groups, new_height, new_width))
        point_maps_groups = np.zeros((n_groups, new_height, new_width))
        for group_name, val in groups_dict.items():
            gi = val[0]
            cluster_list = val[1]
            masks_select = masks[cluster_list]
            point_maps_select = point_maps[cluster_list]
            masks_groups[gi] = masks_select.max(axis=0)
            point_maps_groups[gi] = point_maps_select.max(axis=0)
            io.imsave( os.path.join(out_dir, f"mask_g{gi}_{group_name}.png"), (masks_groups[gi]*255).astype(np.uint8), check_contrast=False)
            io.imsave( os.path.join(out_dir, f"pnts_g{gi}_{group_name}.png"), (point_maps_groups[gi]*255).astype(np.uint8), check_contrast=False)

        # Pad maps
        tmp_map = np.zeros((foreground_mask.shape[0]+padding_width*2, foreground_mask.shape[1]+padding_width*2))
        tmp_map[padding_width:-padding_width, padding_width:-padding_width] = foreground_mask
        foreground_mask_padded = tmp_map

        tmp_map = np.ones((background_mask.shape[0]+padding_width*2, background_mask.shape[1]+padding_width*2))
        tmp_map[padding_width:-padding_width, padding_width:-padding_width] = background_mask
        background_mask_padded = tmp_map

        k = np.ones((3, 3), np.uint8)  # Define 3x3 kernel
        foreground_mask_padded_eroded = cv2.erode(foreground_mask_padded.copy(), k, 1)    
        foreground_mask_padded_dilated = cv2.dilate(foreground_mask_padded.copy(), k, 1)    


        # Compute 0D topology from points
        # print('\ncompute 0d topology from points')
        for group_name, val in groups_dict.items():
            gi = val[0]
            cluster_list = val[1]
            pnt_cloud_name = f"pnts_cc_{group_name}_g{gi}"

            im_cc = point_maps_groups[gi]
            compute_topo_cc(group_name, gi, im_cc, out_dir, foreground_mask_padded, background_mask_padded, new_height, new_width)

            pnt_cloud_name = f"mask_cc_{group_name}_g{gi}"

            im_cc = masks_groups[gi]
            compute_topo_cc(group_name, gi, im_cc, out_dir, foreground_mask_padded, background_mask_padded, new_height, new_width)


        # Compute 1D topology from points - No Convex Hull
        # print('\ncompute 1d topology from points')
        for group_name, val in groups_dict.items():
            gi = val[0]
            cluster_list = val[1]
            pnt_cloud_name = f"pnts_holes_{group_name}_g{gi}"
            print(pnt_cloud_name )

            im_cc = point_maps_groups[gi]

            compute_topo_holes(group_name, gi, im_cc, out_dir, foreground_mask_padded, background_mask_padded, new_height, new_width)

            pnt_cloud_name = f"mask_holes_{group_name}_g{gi}"
            print(pnt_cloud_name )

            im_cc = masks_groups[gi]
            compute_topo_holes(group_name, gi, im_cc, out_dir, foreground_mask_padded, background_mask_padded, new_height, new_width)




        gc.collect()
        with open(done_filepath, "w+") as done_file:
            done_file.write("topo compute done.")
            done_file.flush()
        wi += 1
        sys.stdout.flush()
        