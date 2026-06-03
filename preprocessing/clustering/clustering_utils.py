try:
    import matplotlib
    matplotlib.use('Agg') # set the backend before importing pyplot
    from matplotlib import pyplot as plt
except:
    pass

import warnings
warnings.filterwarnings('ignore')


import os
import sys
import glob
import math
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
import skimage.io as io
from skimage.transform import rescale, resize
try:
    import umap
except:
    pass

cluster_colors5 = {0:'r',1:'g', 2:'b', 3:'c', 4:'m'}
cluster_colors14 = {0:'#6929c4', 
                  1:'#1192e8',
                  2:'#005d5d',
                  3:'#9f1853',
                  4:'#fa4d56',
                  5:'#570408',
                  6:'#198038',
                  7:'#002d9c',
                  8:'#ee538b',
                  9:'#b28600',
                  10:'#009d9a',
                  11:'#012749',
                  12:'#8a3800',
                  13:'#a56eff',
                  14:'#8b1b13', 
                  15:'#234349', 
                  16:'#e8b696', 
                  17:'#50a6c5', 
                  18:'#118098', 
                  19:'#fa8072', 
                  20:'#6effe8', 
                  }

#01. Purple 70
#6929c4
#02. Cyan 50
#1192e8
#03. Teal 70
#005d5d
#04. Magenta 70
#9f1853
#05. Red 50
#fa4d56
#06. Red 90
#570408
#07. Green 60
#198038
#08. Blue 80
#002d9c
#09. Magenta 50
#ee538b
#10. Yellow 50
#b28600
#11. Teal 50
#009d9a
#12. Cyan 90
#012749
#13. Orange 70
#8a3800
#14. Purple 50
#a56eff



def cluster_meanshift(data_arr, bandwidth, data_ids, out_dir, out_suffix, vis):

    clustering = MeanShift(bandwidth=bandwidth).fit(data_arr)
    cluster_centers = clustering.cluster_centers_
    instance_labels = clustering.labels_
    if(out_dir is not None):
        cluster_centers.dump(os.path.join(out_dir,f'ms_clustering{out_suffix}_bw{str(bandwidth)}_centers.npy'))
        instance_labels.dump(os.path.join(out_dir,f'ms_clustering{out_suffix}_bw{str(bandwidth)}_instance_labels.npy'))
        #data_ids.dump(os.path.join(out_dir,f'ms_clustering{out_suffix}_bw{str(bandwidth)}_instance_ids.npy'))
        #id_lbl_pairs = zip(data_ids, cluster_labels)
        df = pd.DataFrame({'slide_id':data_ids, 
                           'cluster_id':instance_labels})
        df.to_csv(os.path.join(out_dir,f'ms_clustering{out_suffix}_bw{str(bandwidth)}_id_lbl_pairs.csv'), sep=',')
        #id_lbl_pairs_arr = np.concatenate((data_ids, cluster_labels), axis=1)
        #np.savetxt(os.path.join(out_dir,f'ms_clustering{out_suffix}_bw{str(bandwidth)}_id_lbl_pairs.txt'), id_lbl_pairs_arr, newline='\n', delimiter=',')
        n_clusters = len(cluster_centers)
        print('n_clusters', n_clusters)
        if(vis == 'tsne'):
            perplexity = min(perplexity, len(instance_labels)//2)
            vis_clustering_tsne(data_arr, instance_labels, n_clusters, out_dir, out_suffix=f'ms_clustering{out_suffix}_bw{str(bandwidth)}')


'''
    Uses: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    alg = "lloyd": the classical EM-style algorithm
    alg = "elkan"  can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters)
'''
def cluster_kmeans(data_arr, n_clusters, data_ids, out_dir, out_suffix, vis, alg="lloyd", pca_init=False, save_distances=True, init_comp=None, perplexity=30):
    if(out_suffix is None):
        out_suffix = ''
    out_suffix = out_suffix.strip()
    if(alg != "lloyd"):
        out_suffix = out_suffix + f"_alg_{alg}"
    if(len(out_suffix) > 0 and out_suffix[0] != '_'):
        out_suffix = '_' + out_suffix
    init = "k-means++"
    if(pca_init):
        pca = PCA(n_components=n_clusters).fit(data_arr)
        init = pca.components_
    #print('init')
    #print(init)
    if(init_comp is not None):
        init=init_comp
    clustering = KMeans(init=init, n_clusters=n_clusters, algorithm=alg).fit(data_arr)
    cluster_centers = clustering.cluster_centers_
    instance_labels = None
    df = None
    if(data_ids is not None):
        instance_labels = clustering.labels_
        df = pd.DataFrame({'slide_id':data_ids, 
                            'cluster_id':instance_labels})
    #print('cluster_centers')
    #print(cluster_centers)
    if(out_dir is not None):
        cluster_centers.dump(os.path.join(out_dir,f'kmeans_clustering{out_suffix}_n{str(n_clusters)}_centers.npy'))
        if(instance_labels is not None):
            instance_labels.dump(os.path.join(out_dir,f'kmeans_clustering{out_suffix}_n{str(n_clusters)}_instance_labels.npy'))
        #data_ids.dump(os.path.join(out_dir,f'kmeans_clustering{out_suffix}_n{str(n_clusters)}_instance_ids.npy'))
        #id_lbl_pairs = zip(data_ids, cluster_labels)
        if(save_distances):
            distances = clustering.transform(data_arr)
            for i in range(n_clusters):
                df[f"dist{i}"] = distances[:,i]
        if(df is not None):
            df.to_csv(os.path.join(out_dir,f'kmeans_clustering{out_suffix}_n{str(n_clusters)}_id_label_pairs.csv'), sep=',', index = False)
        #id_lbl_pairs_arr = np.concatenate((data_ids, cluster_labels), axis=1)
        #np.savetxt(os.path.join(out_dir,f'kmeans_clustering{out_suffix}_n{str(n_clusters)}_id_lbl_pairs.txt'), id_lbl_pairs_arr, newline='\n', delimiter=',')
        n_clusters = len(cluster_centers)
        print('n_clusters', n_clusters)
        if(vis == 'tsne'):
            perplexity = min(perplexity, len(instance_labels)//2)
            vis_clustering_tsne(data_arr, instance_labels, n_clusters, out_dir, out_suffix=f'kmeans_clustering{out_suffix}_n{str(n_clusters)}', perplexity=perplexity)
    return df, cluster_centers


def cluster_spectral(data_arr, n_clusters, data_ids, out_dir, out_suffix, vis):
    if(out_suffix is None):
        out_suffix = ''
    out_suffix = out_suffix.strip()
    if(len(out_suffix) > 0 and out_suffix[0] != '_'):
        out_suffix = '_' + out_suffix
    clustering = SpectralClustering(n_clusters=n_clusters, gamma=5).fit(data_arr)
    #clustering = SpectralClustering(n_clusters=n_clusters, gamma=0, affinity='sigmoid').fit(data_arr)
    #cluster_centers = clustering.cluster_centers_
    instance_labels = clustering.labels_
    df = pd.DataFrame({'slide_id':data_ids, 
                        'cluster_id':instance_labels})
    if(out_dir is not None):
        #cluster_centers.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_centers.npy'))
        instance_labels.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_instance_labels.npy'))
        #data_ids.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_instance_ids.npy'))
        #id_lbl_pairs = zip(data_ids, cluster_labels)
        df.to_csv(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_id_label_pairs.csv'), sep=',', index = False)
        #id_lbl_pairs_arr = np.concatenate((data_ids, cluster_labels), axis=1)
        #np.savetxt(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_id_lbl_pairs.txt'), id_lbl_pairs_arr, newline='\n', delimiter=',')
        #n_clusters = len(cluster_centers)
        print('n_clusters', n_clusters)
        if(vis == 'tsne'):
            perplexity = min(perplexity, len(instance_labels)//2)
            vis_clustering_tsne(data_arr, instance_labels, n_clusters, out_dir, out_suffix=f'spectral_clustering{out_suffix}_n{str(n_clusters)}')
    return df

def cluster_agglomerative_clustering(data_arr, n_clusters, data_ids, out_dir, out_suffix, vis):
    if(out_suffix is None):
        out_suffix = ''
    out_suffix = out_suffix.strip()
    if(len(out_suffix) > 0 and out_suffix[0] != '_'):
        out_suffix = '_' + out_suffix
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", affinity="cosine").fit(data_arr)
    #cluster_centers = clustering.cluster_centers_
    instance_labels = clustering.labels_
    print('data_ids', len(data_ids))
    print('instance_labels', len(instance_labels))
    df = pd.DataFrame({'slide_id':data_ids, 
                        'cluster_id':instance_labels})
    if(out_dir is not None):
        #cluster_centers.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_centers.npy'))
        instance_labels.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_instance_labels.npy'))
        #data_ids.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_instance_ids.npy'))
        #id_lbl_pairs = zip(data_ids, cluster_labels)
        df.to_csv(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_id_label_pairs.csv'), sep=',', index = False)
        #id_lbl_pairs_arr = np.concatenate((data_ids, cluster_labels), axis=1)
        #np.savetxt(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_id_lbl_pairs.txt'), id_lbl_pairs_arr, newline='\n', delimiter=',')
        #n_clusters = len(cluster_centers)
        print('n_clusters', n_clusters)
        if(vis == 'tsne'):
            perplexity = min(perplexity, len(instance_labels)//2)
            vis_clustering_tsne(data_arr, instance_labels, n_clusters, out_dir, out_suffix=f'spectral_clustering{out_suffix}_n{str(n_clusters)}')
    return df

def cluster_dbscan(data_arr, n_clusters, data_ids, out_dir, out_suffix, vis):
    if(out_suffix is None):
        out_suffix = ''
    out_suffix = out_suffix.strip()
    if(len(out_suffix) > 0 and out_suffix[0] != '_'):
        out_suffix = '_' + out_suffix
    clustering = DBSCAN().fit(data_arr)
    #cluster_centers = clustering.cluster_centers_
    instance_labels = clustering.labels_
    print('data_ids', len(data_ids))
    print('instance_labels', len(instance_labels))
    df = pd.DataFrame({'slide_id':data_ids, 
                        'cluster_id':instance_labels})
    if(out_dir is not None):
        #cluster_centers.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_centers.npy'))
        instance_labels.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_instance_labels.npy'))
        #data_ids.dump(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_instance_ids.npy'))
        #id_lbl_pairs = zip(data_ids, cluster_labels)
        df.to_csv(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_id_label_pairs.csv'), sep=',', index = False)
        #id_lbl_pairs_arr = np.concatenate((data_ids, cluster_labels), axis=1)
        #np.savetxt(os.path.join(out_dir,f'spectral_clustering{out_suffix}_n{str(n_clusters)}_id_lbl_pairs.txt'), id_lbl_pairs_arr, newline='\n', delimiter=',')
        #n_clusters = len(cluster_centers)
        print('n_clusters', n_clusters)
        if(vis == 'tsne'):
            perplexity = min(perplexity, len(instance_labels)//2)
            vis_clustering_tsne(data_arr, instance_labels, n_clusters, out_dir, out_suffix=f'spectral_clustering{out_suffix}_n{str(n_clusters)}')
    return df

def cluster_affinity_propagation(data_arr, data_ids, out_dir, out_suffix, vis, n_clusters_preference=None, alg="full", save_distances=True, perplexity=30, max_clusters=math.inf):
# def cluster_affinity_propagation(data_arr, data_ids, out_dir, out_suffix, vis, n_clusters_preference=None, alg="full", perplexity=30):
    if(out_suffix is None):
        out_suffix = ''
    out_suffix = out_suffix.strip()
    if(alg != "full"):
        out_suffix = out_suffix + f"_alg_{alg}"
    if(len(out_suffix) > 0 and out_suffix[0] != '_'):
        out_suffix = '_' + out_suffix
    # init = "k-means++"
    # if(pca_init):
    #     pca = PCA(n_components=n_clusters).fit(data_arr)
    #     init = pca.components_
    # #print('init')
    # #print(init)
    # if(init_comp is not None):
    #     init=init_comp
    print('data_arr', data_arr.shape)
    clustering = AffinityPropagation(random_state=5, preference=n_clusters_preference).fit(data_arr)
    cluster_centers = clustering.cluster_centers_
    print('cluster_centers', cluster_centers.shape)
    n_clusters = cluster_centers.shape[0]
    if(n_clusters > max_clusters ):
        return cluster_kmeans(data_arr, max_clusters, data_ids, out_dir, out_suffix, vis, alg="elkan", pca_init=True, perplexity=5)
    instance_labels = None
    df = None
    if(data_ids is not None):
        instance_labels = clustering.labels_
        df = pd.DataFrame({'slide_id':data_ids, 
                            'cluster_id':instance_labels})
    #print('cluster_centers')
    #print(cluster_centers)
    if(out_dir is not None):
        cluster_centers.dump(os.path.join(out_dir,f'apc_clustering{out_suffix}_n{str(n_clusters)}_pref_{n_clusters_preference}_centers.npy'))
        if(instance_labels is not None):
            instance_labels.dump(os.path.join(out_dir,f'apc_clustering{out_suffix}_n{str(n_clusters)}_pref_{n_clusters_preference}_instance_labels.npy'))
        #data_ids.dump(os.path.join(out_dir,f'apc_clustering{out_suffix}_n{str(n_clusters)}_instance_ids.npy'))
        #id_lbl_pairs = zip(data_ids, cluster_labels)
        if(save_distances):
            # distances = clustering.transform(data_arr)
            distances = euclidean_distances(np.array(cluster_centers), np.array(data_arr)).T
            for i in range(n_clusters):
                df[f"dist{i}"] = distances[:,i]
        if(df is not None):
            df.to_csv(os.path.join(out_dir,f'apc_clustering{out_suffix}_n{str(n_clusters)}_pref_{n_clusters_preference}_id_label_pairs.csv'), sep=',', index = False)
        #id_lbl_pairs_arr = np.concatenate((data_ids, cluster_labels), axis=1)
        #np.savetxt(os.path.join(out_dir,f'apc_clustering{out_suffix}_n{str(n_clusters)}_id_lbl_pairs.txt'), id_lbl_pairs_arr, newline='\n', delimiter=',')
        n_clusters = len(cluster_centers)
        print('n_clusters', n_clusters)
        if(vis == 'tsne'):
            perplexity = min(perplexity, len(instance_labels)//2)
            vis_clustering_tsne(data_arr, instance_labels, n_clusters, out_dir, out_suffix=f'apc_clustering{out_suffix}_n{str(n_clusters)}_pref_{n_clusters_preference}', perplexity=perplexity)
    return df, cluster_centers


def predict_affinity_propagation(cluster_centers_filepath, data_arr, data_ids, out_dir, out_suffix):
    cluster_centers = np.load(cluster_centers_filepath, allow_pickle=True)

    nbrs = NearestNeighbors(n_neighbors=cluster_centers.shape[0], algorithm='brute').fit(cluster_centers)
    distances, indices = nbrs.kneighbors(data_arr)
    print('distances', distances.shape)
    print('indices', indices.shape)
    predicted_labels = indices
    return predicted_labels, distances




def vis_clustering_tsne(data_arr, instance_labels, n_clusters, out_dir, out_suffix, perplexity=30):
    print('data_arr', data_arr.shape)
    print('instance_labels', instance_labels.shape)
    print('n_clusters', n_clusters)
    if(n_clusters <= 5):
        cluster_colors = cluster_colors5
    else:
        cluster_colors = cluster_colors14
    data_2d = TSNE(n_components=2, perplexity=perplexity).fit_transform(data_arr)
    fig, ax = plt.subplots();
    for cci in range(n_clusters):
        Y = data_2d[instance_labels == cci]
        print('Y', Y.shape)
        ax.scatter(Y[:,0], Y[:,1], c=cluster_colors[cci], label=f"cluster_{cci}")
    #ax.axis('tight')
    # Shrink current axis by 20% to fit legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("TSNE")
    plt.savefig(os.path.join(out_dir,f'{out_suffix}_vis_tsne.png'))    
    plt.close(fig)

def vis_clustering_umap(data_arr, instance_labels, n_clusters, out_dir, out_suffix, n_neighbors=15, metric='euclidean'):
    print('data_arr', data_arr.shape)
    print('instance_labels', instance_labels.shape)
    print('n_clusters', n_clusters)
    if(n_clusters <= 5):
        cluster_colors = cluster_colors5
    else:
        cluster_colors = cluster_colors14
    print('before fit_transform')
    data_2d = umap.UMAP(n_neighbors=n_neighbors, metric=metric).fit_transform(data_arr)
    print('after fit_transform')
    fig, ax = plt.subplots();
    for cci in range(n_clusters):
        Y = data_2d[instance_labels == cci]
        print('Y', Y.shape)
        ax.scatter(Y[:,0], Y[:,1], c=cluster_colors[cci], label=f"cluster_{cci}")
    #ax.axis('tight')
    # Shrink current axis by 20% to fit legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("UMAP")
    plt.savefig(os.path.join(out_dir,f'{out_suffix}_vis_umap.png'))    
    plt.close(fig)

def load_clustering():
    return

def load_survival():
    return

def get_clusters_representative_samples(df_clustering_data, slide_id_col, cluster_id_col, distance_columns, return_n_per_cluster):
    nc = len(distance_columns)
    cluster_samples_dict = {}
    for ci in range(nc):
        print('ci', ci)
        print('distance_columns[ci]', distance_columns[ci])
        df_cluster_ci = df_clustering_data[df_clustering_data[cluster_id_col]==ci]
        # print('df_cluster_ci.head(return_n_per_cluster)[slide_id_col]')
        print(df_cluster_ci.head(return_n_per_cluster))
        df_cluster_ci = df_cluster_ci.sort_values(by=distance_columns[ci], ascending=True)
        print('df_cluster_ci.head(return_n_per_cluster)[slide_id_col]')
        print(df_cluster_ci.head(return_n_per_cluster))
        rep_samples = df_cluster_ci.head(return_n_per_cluster)[slide_id_col]  # todo check number of rows in df >= return_n_per_cluster
        cluster_samples_dict[ci] = rep_samples.tolist()
        #df_cluster_ci.to_csv(os.path.join('/mnt/data05/shared/sabousamra/spatial_analysis/tcga_brca_all_single_nonfat_k150_v4_opt2/debug_cluster_vis',f'nc{nc}_c{ci}_id_label_pairs.csv'), sep=',', index = False)

    return cluster_samples_dict


def vis_samples_in_single_figure(data_dir, filepattern, slide_ids, resize_fraction, resize_fixed_height, n_per_row, out_dir, out_file_suffix, padding = 30, bg_color=255):
    print('in vis_samples_in_single_figure')
    print('slide_ids', slide_ids)
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    # load the images
    img_list = []
    print('data_dir', data_dir)
    for i in range(len(slide_ids)):
        print('slide_ids[i]', slide_ids[i])
        print("os.path.join(data_dir, filepattern.replace('{slide_id}', slide_ids[i]))", os.path.join(data_dir, filepattern.replace('<slide_id>', slide_ids[i])))
        
        slide_filepattern = filepattern.replace('{slide_id}', slide_ids[i]).replace('<slide_id>', slide_ids[i])
        slide_filepath = os.path.join(data_dir, filepattern.replace('{slide_id}', slide_ids[i]).replace('<slide_id>', slide_ids[i]))
        sample_img = None
        if(not os.path.exists(slide_filepath)):
            if('*' in slide_filepath):
                files = glob.glob(slide_filepath)
                if(files is None or len(files) == 0):
                    sample_img = np.zeros((100,100,3))
                else:
                    slide_filepath = files[0]
            else:
                sample_img = np.zeros((100,100,3))
        if(sample_img is None):
            print(slide_filepath)
            sample_filepath = slide_filepath
            sample_img = io.imread(sample_filepath)
            if(len(sample_img.shape)>2 and sample_img.shape[-1]>3):
                sample_img = sample_img[:,:,:3]
            if(resize_fraction is not None and resize_fraction > 0):
                sample_img = resize(sample_img, (sample_img.shape[0]*resize_fraction, sample_img.shape[1]*resize_fraction), preserve_range=True)
            if(resize_fixed_height > 0):
                resize_fraction = resize_fixed_height / sample_img.shape[0]
                print('resize_fraction', resize_fraction)
                sample_img = resize(sample_img, (sample_img.shape[0]*resize_fraction, sample_img.shape[1]*resize_fraction), preserve_range=True)
            if (len(sample_img.shape)==2): # expand grayscale image to three channel.
                sample_img=sample_img[:,:,np.newaxis]
                sample_img=np.concatenate((sample_img,sample_img,sample_img),2)

        img_list.append(sample_img)

    # compute the width and height needed
    vis_img_width = 0
    vis_img_height = 0
    for ri in range((len(slide_ids)+n_per_row-1)//n_per_row):
        row_width = 0
        row_height = 0
        for i in range(ri*n_per_row, min((ri+1)*n_per_row, len(slide_ids))):
            row_width += padding
            row_width += img_list[i].shape[1]
            row_height = max(img_list[i].shape[0], row_height)
        row_width += padding
        vis_img_width = max(vis_img_width, row_width)
        vis_img_height += padding
        vis_img_height += row_height
    vis_img_height += padding

    if(len(img_list[0].shape) == 3):
        vis_img = np.ones((vis_img_height, vis_img_width, 3))*bg_color
    else:
        vis_img = np.ones((vis_img_height, vis_img_width))*bg_color

    pos_y = 0
    pos_x = 0
    for ri in range((len(slide_ids)+n_per_row-1)//n_per_row):
        print('ri', ri)
        pos_x = 0
        pos_y += padding
        row_height = 0
        for i in range(ri*n_per_row, min((ri+1)*n_per_row, len(slide_ids))):
            print('i', i)
            pos_x += padding
            vis_img[pos_y:pos_y+img_list[i].shape[0], pos_x:pos_x+img_list[i].shape[1]] = img_list[i]
            pos_x += img_list[i].shape[1]
            row_height = max(row_height, img_list[i].shape[0])
        pos_y += row_height

    io.imsave(os.path.join(out_dir, f"{out_file_suffix}.png"), vis_img.astype(np.uint8), check_contrast=False)
    with open(os.path.join(out_dir, f"{out_file_suffix}.txt"), 'w+') as out_file:
        for i in range(len(slide_ids)):
            out_file.write(f"{slide_ids[i]}\n")
            out_file.flush()

    #return