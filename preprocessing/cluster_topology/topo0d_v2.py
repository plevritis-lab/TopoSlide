import numpy as np


INDEXING_Y = 0
INDEXING_X = 1
INDEXING_VAL = 2
INDEXING_PARENT_Y = 3
INDEXING_PARENT_X = 4
INDEXING_BIRTH_Y = 5
INDEXING_BIRTH_X = 6
INDEXING_DEATH_Y = 7
INDEXING_DEATH_X = 8
INDEXING_LEN = 9
INDEXING_PROCESSED = 10


def compute_persistence_2DImg_0DHom_single_pad2(likelihood, connectivity, padwidth, pad_value=None, death_thresh=0,name_id=None):
    input_height = likelihood.shape[0]
    input_width = likelihood.shape[1]
    if(padwidth > 0):
        if(pad_value is None):
            pad_value = likelihood.max()
        likelihood_tmp = np.ones((likelihood.shape[0]+padwidth*2, likelihood.shape[1]+padwidth*2)) * pad_value
        likelihood_tmp[padwidth:-padwidth, padwidth:-padwidth] = likelihood
        likelihood = likelihood_tmp      
        

    pixels = np.zeros((likelihood.shape[0], likelihood.shape[1], 11))
    
    pixels_sorted = np.zeros((likelihood.shape[0]* likelihood.shape[1]),dtype = [('y', int), ('x', int), ('val', float)])

    # init pixels
    i = -1
    for y in range(likelihood.shape[0]):
        for x in range(likelihood.shape[1]):
            pixels[y,x,INDEXING_Y] = y 
            pixels[y,x,INDEXING_X] = x
            pixels[y,x,INDEXING_VAL] = likelihood[y,x]
            pixels[y,x,INDEXING_PARENT_Y] = -1
            pixels[y,x,INDEXING_PARENT_X] = -1
            pixels[y,x,INDEXING_BIRTH_Y] = y
            pixels[y,x,INDEXING_BIRTH_X] = x
            pixels[y,x,INDEXING_DEATH_Y] = -1
            pixels[y,x,INDEXING_DEATH_X] = -1
            pixels[y,x,INDEXING_LEN] = 1
            pixels[y,x,INDEXING_PROCESSED] = 0
    
            i += 1
            pixels_sorted[i]['y'] = y
            pixels_sorted[i]['x'] = x
            pixels_sorted[i]['val'] = likelihood[y,x]


    pixels_sorted = np.sort(pixels_sorted, order='val')[::-1]
    
    for pixel_tmp in pixels_sorted:
        # connect new pixel to first component near it
        parent = None
        y = pixel_tmp[INDEXING_Y]
        x = pixel_tmp[INDEXING_X]
        pixel = pixels[y,x]
        parent_node = None # assign to object with largest peak (earliest born)
        parent_bcp_val = None
        if(y-1 >= 0 and pixels[y-1,x, INDEXING_PROCESSED] > 0):
            p1 = find(pixels, pixels[y-1,x])
            p1_bcp_val = pixels[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]
            if(parent_node is None or parent_bcp_val < p1_bcp_val):
                parent_node = p1
                parent_bcp_val = p1_bcp_val
                parent = pixels[y-1,x]
                p_y = y-1
                p_x = x
        if(y+1 < likelihood.shape[0] and pixels[y+1,x, INDEXING_PROCESSED] > 0):
            p1 = find(pixels, pixels[y+1,x])
            p1_bcp_val = pixels[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]
            if(parent_node is None or parent_bcp_val < p1_bcp_val):
                parent_node = p1
                parent_bcp_val = p1_bcp_val
                parent = pixels[y+1,x]
                p_y = y+1
                p_x = x
        if(x-1 >= 0 and pixels[y,x-1, INDEXING_PROCESSED] > 0):
            p1 = find(pixels, pixels[y,x-1])
            p1_bcp_val = pixels[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]
            if(parent_node is None or parent_bcp_val < p1_bcp_val):
                parent_node = p1
                parent_bcp_val = p1_bcp_val
                parent = pixels[y,x-1]
                p_y = y
                p_x = x-1
        if(x+1 < likelihood.shape[1] and pixels[y,x+1, INDEXING_PROCESSED] > 0):
            p1 = find(pixels, pixels[y,x+1])
            p1_bcp_val = pixels[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]
            if(parent_node is None or parent_bcp_val < p1_bcp_val):
                parent_node = p1
                parent_bcp_val = p1_bcp_val
                parent = pixels[y,x+1]
                p_y = y
                p_x = x+1
        if((connectivity == 8) and y-1 >= 0 and x-1 >= 0 and pixels[y-1,x-1, INDEXING_PROCESSED] > 0):
            p1 = find(pixels, pixels[y-1,x-1])
            p1_bcp_val = pixels[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]
            if(parent_node is None or parent_bcp_val < p1_bcp_val):
                parent_node = p1
                parent_bcp_val = p1_bcp_val
                parent = pixels[y-1,x-1]
                p_y = y-1
                p_x = x-1
        if((connectivity == 8) and y-1 >= 0 and x+1 < likelihood.shape[1] and pixels[y-1,x+1, INDEXING_PROCESSED] > 0):
            p1 = find(pixels, pixels[y-1,x+1])
            p1_bcp_val = pixels[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]
            if(parent_node is None or parent_bcp_val < p1_bcp_val):
                parent_node = p1
                parent_bcp_val = p1_bcp_val
                parent = pixels[y-1,x+1]
                p_y = y-1
                p_x = x+1
        if((connectivity == 8) and y+1 < likelihood.shape[0] and x+1 < likelihood.shape[1] and pixels[y+1,x+1, INDEXING_PROCESSED] > 0):
            p1 = find(pixels, pixels[y+1,x+1])
            p1_bcp_val = pixels[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]
            if(parent_node is None or parent_bcp_val < p1_bcp_val):
                parent_node = p1
                parent_bcp_val = p1_bcp_val
                parent = pixels[y+1,x+1]
                p_y = y+1
                p_x = x+1
        if((connectivity == 8) and y+1 < likelihood.shape[0] and x-1 >= 0 and pixels[y+1,x-1, INDEXING_PROCESSED] > 0):
            p1 = find(pixels, pixels[y+1,x-1])
            p1_bcp_val = pixels[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]
            if(parent_node is None or parent_bcp_val < p1_bcp_val):
                parent_node = p1
                parent_bcp_val = p1_bcp_val
                parent = pixels[y+1,x-1]
                p_y = y+1
                p_x = x-1

        if(not(parent_node is None)):
            parent = union_b(pixels, parent_node, pixel)
        # add the other edges
        if(y+1 < likelihood.shape[0] and (pixels[y+1,x, INDEXING_PROCESSED] > 0) and not(p_y==y+1 and p_x ==x )):
            union_d(pixels, parent, pixels[y+1,x],pixel, death_thresh)
        if(x-1 >= 0 and (pixels[y,x-1, INDEXING_PROCESSED] > 0) and not(p_y==y and p_x ==x-1 )):
            union_d(pixels, parent, pixels[y,x-1], pixel, death_thresh)
        if(x+1 < likelihood.shape[1] and (pixels[y,x+1, INDEXING_PROCESSED] > 0) and not(p_y==y and p_x ==x+1 )):
            union_d(pixels, parent, pixels[y,x+1], pixel, death_thresh)
        if(connectivity == 8):
            if(y-1 >= 0 and x-1 >= 0 and pixels[y-1,x-1, INDEXING_PROCESSED] > 0 and not(p_y==y-1 and p_x ==x-1 )):
                union_d(pixels, parent, pixels[y-1,x-1],pixel, death_thresh)
            if(y-1 >= 0 and x+1 < likelihood.shape[1] and pixels[y-1,x+1, INDEXING_PROCESSED] > 0 and not(p_y==y-1 and p_x ==x+1 )):
                union_d(pixels, parent, pixels[y-1,x+1],pixel, death_thresh)
            if(y+1 < likelihood.shape[0] and x+1 < likelihood.shape[1] and pixels[y+1,x+1, INDEXING_PROCESSED] > 0 and not(p_y==y+1 and p_x ==x+1 )):
                union_d(pixels, parent, pixels[y+1,x+1], pixel, death_thresh)
            if(y+1 < likelihood.shape[0] and x-1 >= 0 and pixels[y+1,x-1, INDEXING_PROCESSED] > 0 and not(p_y==y+1 and p_x ==x-1 )):
                union_d(pixels, parent, pixels[y+1,x-1], pixel, death_thresh)

        pixels[y,x, INDEXING_PROCESSED] = 1

    pixels_1d = pixels.reshape((-1,11))
    comps = pixels_1d[:,INDEXING_DEATH_Y] > 0
    cps = pixels_1d[np.where(comps>0)]
    pd = np.zeros((cps.shape[0],2))
    bcp = np.zeros((cps.shape[0],2))
    dcp = np.zeros((cps.shape[0],2))    
    i = -1
    for node in cps:
        i += 1
        birth_cp = pixels[int(node[INDEXING_BIRTH_Y]), int(node[INDEXING_BIRTH_X])]
        death_cp = pixels[int(node[INDEXING_DEATH_Y]), int(node[INDEXING_DEATH_X])]
        pd[i,0]=death_cp[INDEXING_VAL] # val of death cp
        pd[i,1]=birth_cp[INDEXING_VAL] # val of birth cp
        bcp[i,0]= node[INDEXING_BIRTH_Y]
        bcp[i,1]= node[INDEXING_BIRTH_X]
        dcp[i,0]= node[INDEXING_DEATH_Y]
        dcp[i,1]= node[INDEXING_DEATH_X]

    bcp = bcp - padwidth
    dcp = dcp - padwidth
    select_cps_bcp1 = np.logical_and(bcp[:,0]>=0, bcp[:,1]>=0)
    select_cps_bcp2 = np.logical_and(bcp[:,0]<input_height, bcp[:,1]<input_width)
    select_cps = np.logical_and(select_cps_bcp1, select_cps_bcp2)
    pd = pd[select_cps]
    bcp = bcp[select_cps]
    dcp = dcp[select_cps]

    del pixels, pixels_sorted
    
    return pd, bcp, dcp

def find(nodes, node1): 

    while(node1[INDEXING_PARENT_Y] >= 0):  # while has parent
        node1 = find(nodes, nodes[int(node1[INDEXING_PARENT_Y]),int(node1[INDEXING_PARENT_X])])
    return node1


# union of component and a new pixel
def union_b(nodes, comp1, pixel1):

    node1=comp1
    node2=pixel1
    p1 = find(nodes, node1)
    p2 = find(nodes, node2)

    # length check
    if(p1[INDEXING_LEN] < p2[INDEXING_LEN]):
        # swap p1 and p2
        tmp = p1
        p1 = p2
        p2 = tmp
    elif(p1[INDEXING_LEN] == p2[INDEXING_LEN]):
        nodes[int(p1[INDEXING_Y]), int(p1[INDEXING_X]), INDEXING_LEN]  += 1
        
    # update parent
    nodes[int(p2[INDEXING_Y]), int(p2[INDEXING_X]), INDEXING_PARENT_Y]  = p1[INDEXING_Y]
    nodes[int(p2[INDEXING_Y]), int(p2[INDEXING_X]), INDEXING_PARENT_X]  = p1[INDEXING_X]


    # update birth
    nodes[int(p2[INDEXING_Y]), int(p2[INDEXING_X]), INDEXING_BIRTH_Y]  = p1[INDEXING_BIRTH_Y]
    nodes[int(p2[INDEXING_Y]), int(p2[INDEXING_X]), INDEXING_BIRTH_X]  = p1[INDEXING_BIRTH_X]

    return p1



# union of 2 components (will involve a saddle point)
def union_d(nodes, comp1, comp2, saddle, death_thresh=0):
    node1=comp1
    node2=comp2
    p1 = find(nodes, node1)
    p2 = find(nodes, node2)
    if(p1[INDEXING_Y] == p2[INDEXING_Y] and p1[INDEXING_X] == p2[INDEXING_X]): # they have the same parent. i.e. already connected
        return p1

    # birh time check
    if(nodes[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL] < nodes[int(p2[INDEXING_BIRTH_Y]), int(p2[INDEXING_BIRTH_X]), INDEXING_VAL]):
        # swap p1 and p2
        tmp = p1
        p1 = p2
        p2 = tmp

    # update length
    nodes[int(p1[INDEXING_Y]), int(p1[INDEXING_X]), INDEXING_LEN]  =  max(p1[INDEXING_LEN], p2[INDEXING_LEN]+1)
        
    # update parent
    nodes[int(p2[INDEXING_Y]), int(p2[INDEXING_X]), INDEXING_PARENT_Y]  = p1[INDEXING_PARENT_Y]
    nodes[int(p2[INDEXING_Y]), int(p2[INDEXING_X]), INDEXING_PARENT_X]  = p1[INDEXING_PARENT_X]

    if(abs(node1[INDEXING_VAL] - node2[INDEXING_VAL]) >= death_thresh): # if the pixels at the edge end points have equal value then no death yet
        # update death if don't have death cp
        if(nodes[int(p2[INDEXING_BIRTH_Y]), int(p2[INDEXING_BIRTH_X]), INDEXING_VAL]  <= nodes[int(p1[INDEXING_BIRTH_Y]), int(p1[INDEXING_BIRTH_X]), INDEXING_VAL]  and p2[INDEXING_DEATH_Y] < 0):
            nodes[int(p2[INDEXING_Y]), int(p2[INDEXING_X]), INDEXING_DEATH_Y]  =  saddle[INDEXING_Y]
            nodes[int(p2[INDEXING_Y]), int(p2[INDEXING_X]), INDEXING_DEATH_X]  =   saddle[INDEXING_X]

    return p1



