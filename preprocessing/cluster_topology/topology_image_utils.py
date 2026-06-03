import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot

import sys
sys.path.append("..")

import os
import glob
import math

from matplotlib import pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from scipy.spatial import KDTree
from skimage.draw import line

from topo0d_v2 import compute_persistence_2DImg_0DHom_single_pad2

    

def read_coord_scaled_csv(csv_filepath, scale, height, width):
    new_height = int(height*scale)
    new_width = int(width*scale)
    coord_classes = np.loadtxt(csv_filepath, dtype=int)            
    if(len(coord_classes.shape)==1):
        coord_classes = coord_classes.reshape((1,-1))
    coord_classes[:,0:2] = coord_classes[:,0:2]*scale
    coord_y = coord_classes[:,1].copy()
    coord_x = coord_classes[:,0].copy()
    coord_y[coord_y >= new_height] = new_height-1
    coord_x[coord_x >= new_width] = new_width-1
    coord_classes[:,0] = coord_y 
    coord_classes[:,1] = coord_x
    return coord_classes.astype(int)


def compute_distance_transform(height, width, coord_y, coord_x, out_dir=None, out_file_basename=None ):
    im = np.ones((height, width))
    im[(coord_y, coord_x)] = 0
    im_dist = ndimage.distance_transform_edt(im.astype(float)) # computes euclidean distances on each TRUE position to the nearest background FALSE position
    if(out_dir and out_file_basename):
        io.imsave(os.path.join(out_dir, f"{out_file_basename}_dist.png"), (im_dist/im_dist.max()*255).astype(np.uint8), check_contrast=False)
        im_dist.astype(np.float16).dump(os.path.join(out_dir, f"{out_file_basename}_dist.npy"))
    return im_dist


def vis_save_topo(pd, bcp, dcp, pers, height, width, out_dir, out_file_basename ):
    if(len(bcp) == 0):
        return
    topo_vis_bcp = np.zeros((height, width))
    topo_vis_dcp = np.zeros((height, width))
    topo_vis_bcp[(bcp[:,0], bcp[:,1])] = pers
    topo_vis_dcp[(dcp[:,0], dcp[:,1])] = pers
    if(out_file_basename is not None):
        io.imsave(os.path.join(out_dir, f"{out_file_basename}_topo_bcp.png"), (topo_vis_bcp/topo_vis_bcp.max()*255).astype(np.uint8), check_contrast=False)
        io.imsave(os.path.join(out_dir, f"{out_file_basename}_topo_dcp.png"), (topo_vis_dcp/topo_vis_dcp.max()*255).astype(np.uint8), check_contrast=False)
    topo_vis_bcp_binary = np.zeros((height, width))
    topo_vis_dcp_binary = np.zeros((height, width))
    topo_vis_bcp_binary[(bcp[:,0], bcp[:,1])] = 1
    topo_vis_dcp_binary[(dcp[:,0], dcp[:,1])] = 1
    if(out_file_basename is not None):
        io.imsave(os.path.join(out_dir, f"{out_file_basename}_topo_bcp_binary.png"), (topo_vis_bcp_binary/topo_vis_bcp_binary.max()*255).astype(np.uint8), check_contrast=False)
        io.imsave(os.path.join(out_dir, f"{out_file_basename}_topo_dcp_binary.png"), (topo_vis_dcp_binary/topo_vis_dcp_binary.max()*255).astype(np.uint8), check_contrast=False)
    pd.dump(os.path.join(out_dir, f"{out_file_basename}_topo_pd.npy"))
    bcp.dump(os.path.join(out_dir, f"{out_file_basename}_topo_bcp.npy"))
    dcp.dump(os.path.join(out_dir, f"{out_file_basename}_topo_dcp.npy"))
    pers.dump(os.path.join(out_dir, f"{out_file_basename}_topo_pers.npy"))
    topo_vis_cp_pair = np.zeros((height, width))
    topo_vis_bcp[(bcp[:,0], bcp[:,1])] = 1
    topo_vis_bcp[(dcp[:,0], dcp[:,1])] = 1
    for i in range(len(bcp[:,0])):
        rr, cc = line(bcp[i,0], bcp[i,1], dcp[i,0], dcp[i,1])
        topo_vis_cp_pair[rr, cc] = 1
    if(out_file_basename is not None):
        io.imsave(os.path.join(out_dir, f"{out_file_basename}_topo_cp_pair.png"), (topo_vis_cp_pair*255).astype(np.uint8), check_contrast=False)


def compute_topo(im,thresh=0, padwidth=1, pad_value=None, name_id=None):
    height = im.shape[0]
    width = im.shape[1]
    if(pad_value is None):
        pad_value = im.min()
    pd, bcp, dcp = compute_persistence_2DImg_0DHom_single_pad2(im, 8, padwidth = padwidth, pad_value=pad_value,death_thresh=thresh,name_id=name_id)
    if(len(pd)<=0):
        pers = []
    else:
        bcp[bcp<0] = 0
        dcp[dcp<0] = 0
        bcp = bcp.astype(int)
        dcp = dcp.astype(int)
        bcp_y = bcp[:,0]
        bcp_x = bcp[:,1]
        dcp_y = dcp[:,0]
        dcp_x = dcp[:,1]
        bcp_y[bcp_y>= height] = height-1
        bcp_x[bcp_x>= width] = width-1
        dcp_y[dcp_y>= height] = height-1
        dcp_x[dcp_x>= width] = width-1
        bcp[:,0] = bcp_y
        bcp[:,1] = bcp_x
        dcp[:,0] = dcp_y
        dcp[:,1] = dcp_x
        pers = pd[:,1] - pd[:,0]
        bcp = bcp[pers>0]
        dcp = dcp[pers>0]
        pd = pd[pers>0]
        pers = pers[pers>0]
    return (pd, bcp, dcp, pers)



def plot_pers_diag(bcp_val, dcp_val, out_dir, out_file_basename, max_val=None, out_file_suffix="", plt_font_size=15, plt_marker_size=70):
    fig, ax = plt.subplots(1)
    plt.xlabel("birth", fontweight ='bold', fontsize = plt_font_size)
    plt.ylabel("death", fontweight ='bold', fontsize = plt_font_size)
    plt.xticks(fontsize=plt_font_size)
    plt.yticks(fontsize=plt_font_size)
    plt.title("Persistence Diagram ", fontweight ='bold', fontsize = plt_font_size)
    ax.scatter(bcp_val, dcp_val, s=plt_marker_size)
    pers = bcp_val - dcp_val
    if(pers.max() < 0):
        pers = -pers
    n_ticks = 10
    if(bcp_val.min() >= 0 and dcp_val.min() >= 0):
        min_val = 0
        if(max_val is not None):
            tick_step = max_val/n_ticks
        else:
            tick_step = math.ceil(max(bcp_val.max(), dcp_val.max())/n_ticks)
    else:
        min_val = math.floor(min(bcp_val.min(), dcp_val.min()))
        if(max_val is not None):
            tick_step = (max_val-min_val)/n_ticks
        else:
            tick_step = math.ceil((max(bcp_val.max(), dcp_val.max()) - min(bcp_val.min(), dcp_val.min()))/n_ticks)
    print('plot_pers_diag: min_val, max_val, n_ticks, tick_step', min_val, max_val, n_ticks, tick_step)

    plt.xticks([float(f"{(min_val+tick_step*r):.2f}") for r in range(0, n_ticks+1)], [float(f"{(min_val+tick_step*r):.2f}") for r in range(0, n_ticks+1)])
    plt.yticks([float(f"{(min_val+tick_step*r):.2f}") for r in range(0, n_ticks+1)], [float(f"{(min_val+tick_step*r):.2f}") for r in range(0, n_ticks+1)])
    
    ax.set_ylim(bottom=min_val) 
    ax.set_xlim(left=min_val)
    ax.set_aspect('equal') # make x and y axis same scale
    plt.axline((min_val,min_val), slope=1, color='black', linestyle="--") # draw diagonal, the x==y line
    plt.axline((0,min_val), slope=math.inf, color='black', linestyle="--") # draw vertical line, the x=0 line
    plt.axline((min_val, 0), slope=0, color='black', linestyle="--") # draw horizontal line, the y=0 line
    plt.gcf().set_size_inches((20, 20))    
    fig.savefig(os.path.join(out_dir, f"{out_file_basename}_pers_plot{out_file_suffix}.png"));
    plt.close(fig)
    plt.clf()
    return 


def compute_dtm_k_frac(k_frac, height, width, coord_y, coord_x, out_dir=None, out_file_basename=None ):
    k = k_frac * len(coord_y)
    return compute_dtm_k_fixed(k, height, width, coord_y, coord_x, out_dir=out_dir, out_file_basename=out_file_basename )

# dtm = distance_to_measure, average of distance to k nearest neighbors
def compute_dtm_k_fixed(k, height, width, coord_y, coord_x, out_dir=None, out_file_basename=None ):
    im = np.ones((height, width))
    im[(coord_y, coord_x)] = 0
    (query_points_y, query_points_x) = np.where(im==1)
    query_points = np.zeros((len(query_points_x), 2))
    query_points[:,0] = query_points_y
    query_points[:,1] = query_points_x

    # build kdtree
    leafsize = 2048
    points = np.zeros((len(coord_y),2))
    points[:,0] = coord_y
    points[:,1] = coord_x
    tree = KDTree(points, leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(query_points, k=k)
    dtm = distances.mean(axis=-1)
    im_dist = np.zeros(im.shape)
    im_dist[(query_points_y, query_points_x)] = dtm

    #im_dist = ndimage.distance_transform_edt(im.astype(float)) # computes euclidean distances on each TRUE position to the nearest background FALSE position
    if(out_dir and out_file_basename):
        io.imsave(os.path.join(out_dir, f"{out_file_basename}_im.png"), ((im)*255).astype(np.uint8), check_contrast=False)
        im.astype(np.uint8).dump(os.path.join(out_dir, f"{out_file_basename}_im.npy"))
        io.imsave(os.path.join(out_dir, f"{out_file_basename}_dtm_k{k}.png"), (im_dist/im_dist.max()*255).astype(np.uint8), check_contrast=False)
        im_dist.astype(np.float16).dump(os.path.join(out_dir, f"{out_file_basename}_dtm_k{k}.npy"))
    return im_dist



